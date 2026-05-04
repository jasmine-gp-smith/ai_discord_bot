import os
import asyncio
import httpx
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from north_mcp_python_sdk import NorthMCPServer
import discord

# ---------------------- ENV / SERVER ----------------------

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_TOKEN missing in .env")

THECATAPI_KEY = os.getenv("THECATAPI_KEY")
if not THECATAPI_KEY:
    raise RuntimeError("THECATAPI_KEY missing in .env")

mcp = NorthMCPServer(name="Discord Bot", host="0.0.0.0", port=3001)

# One Discord client for everything
intents = discord.Intents.default()
intents.guilds = True
intents.members = True  # enable "Server Members Intent" in Developer Portal
intents.messages = True  # useful for history-related operations
client = discord.Client(intents=intents)

_ready = asyncio.Event()
_start_lock = asyncio.Lock()
_started = False

DISCORD_READY_TIMEOUT_SECONDS = 20


# ---------------------- SMALL UTILS ----------------------

def _iso_to_discord(dt_iso: str) -> str:
    return dt_iso


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _strip_mention(s: str) -> str:
    """
    Supports <@123>, <@!123>, <@&123> and returns the numeric part as string.
    Also returns raw numeric strings unchanged.
    """
    s = (s or "").strip()
    if s.startswith("<@") and s.endswith(">"):
        s = s[2:-1]
        if s.startswith(("!", "&")):
            s = s[1:]
    return s


def _is_int_string(s: str) -> bool:
    return bool(s) and s.isdigit()


@client.event
async def on_ready():
    _ready.set()
    print(f"Discord logged in as {client.user}.")


async def ensure_discord_started() -> None:
    """
    Ensure the Discord client is started exactly once and ready before tool use.
    Adds a timeout so tool calls don't hang forever if login fails.
    """
    global _started
    async with _start_lock:
        if not _started:
            _started = True
            asyncio.create_task(client.start(DISCORD_TOKEN))

    try:
        await asyncio.wait_for(_ready.wait(), timeout=DISCORD_READY_TIMEOUT_SECONDS)
    except asyncio.TimeoutError as e:
        raise RuntimeError(
            f"Discord client did not become ready within "
            f"{DISCORD_READY_TIMEOUT_SECONDS}s. Check token/network/intents."
        ) from e


async def _get_guild(guild_id: int) -> discord.Guild:
    await ensure_discord_started()
    guild = client.get_guild(guild_id)
    if guild is None:
        guild = await client.fetch_guild(guild_id)
    return guild


# ---------------------- CHANNEL HELPERS ----------------------

async def resolve_text_channel_id(guild_id: int, channel_ref: str) -> int:
    """
    channel_ref can be:
      - numeric ID as string
      - '#name' or 'name'
    """
    guild = await _get_guild(guild_id)

    ref = (channel_ref or "").strip()
    if not ref:
        raise ValueError("channel_ref cannot be empty")

    maybe_id = _strip_mention(ref)
    if _is_int_string(maybe_id):
        return int(maybe_id)

    name = _norm(ref).lstrip("#")
    for ch in guild.text_channels:
        if _norm(ch.name) == name:
            return ch.id

    raise ValueError(f"Couldn't find text channel named #{name}")


# ---------------------- ROLE HELPERS ----------------------

async def find_roles(
    guild_id: int, role_query: str, limit: int = 10
) -> List[Dict[str, Any]]:
    guild = await _get_guild(guild_id)

    rq_raw = str(role_query).strip() if role_query is not None else ""
    rq = _norm(rq_raw).lstrip("@")
    if not rq:
        raise ValueError("role_query cannot be empty")

    maybe_id = _strip_mention(rq_raw)
    if _is_int_string(maybe_id):
        role_id = int(maybe_id)
        role = guild.get_role(role_id)
        if role is None:
            roles = await guild.fetch_roles()
            role = next((r for r in roles if r.id == role_id), None)
        return [{"id": role.id, "name": role.name}] if role else []

    matches = [r for r in guild.roles if _norm(r.name) == rq]

    if not matches:
        roles = await guild.fetch_roles()
        matches = [r for r in roles if _norm(r.name) == rq]

    if not matches:
        roles = await guild.fetch_roles()
        matches = [r for r in roles if rq in _norm(r.name)]

    matches = matches[:limit]
    return [{"id": r.id, "name": r.name} for r in matches]


async def resolve_role_id(guild_id: int, role_query: str) -> int:
    roles = await find_roles(guild_id, role_query, limit=10)
    if not roles:
        raise ValueError(f"Role '{role_query}' not found")

    if len(roles) > 1:
        preview = ", ".join(f"{r['name']} (id={r['id']})" for r in roles[:5])
        raise ValueError(
            f"Ambiguous role '{role_query}'. Matches: {preview}. "
            "Please provide a more specific role name or the role ID."
        )

    return int(roles[0]["id"])


# ---------------------- USER / MEMBER HELPERS ----------------------

async def find_members(
    guild_id: int,
    user_query: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    guild = await _get_guild(guild_id)

    uq_raw = str(user_query).strip() if user_query is not None else ""
    if not uq_raw:
        raise ValueError("user_query cannot be empty")

    maybe_id = _strip_mention(uq_raw)
    if _is_int_string(maybe_id):
        user_id = int(maybe_id)
        try:
            m = guild.get_member(user_id) or await guild.fetch_member(user_id)
            return [
                {
                    "id": m.id,
                    "username": m.name,
                    "global_name": getattr(m, "global_name", None),
                    "display_name": m.display_name,
                }
            ]
        except discord.NotFound:
            return []

    qn = _norm(uq_raw)
    matches: List[discord.Member] = []

    for m in getattr(guild, "members", []):
        if (
            _norm(m.name) == qn
            or _norm(m.display_name) == qn
            or _norm(getattr(m, "global_name", "") or "") == qn
        ):
            matches.append(m)

    if not matches:
        try:
            queried = await guild.query_members(query=uq_raw, limit=limit)
            matches.extend(queried)
        except discord.Forbidden:
            pass

    if not matches and len(uq_raw) >= 3:
        try:
            queried = await guild.query_members(query=uq_raw[:3], limit=limit)
            for m in queried:
                if (
                    qn in _norm(m.name)
                    or qn in _norm(m.display_name)
                    or qn in _norm(getattr(m, "global_name", "") or "")
                ):
                    matches.append(m)
        except discord.Forbidden:
            pass

    seen = set()
    uniq: List[discord.Member] = []
    for m in matches:
        if m.id not in seen:
            seen.add(m.id)
            uniq.append(m)

    uniq = uniq[:limit]
    return [
        {
            "id": m.id,
            "username": m.name,
            "global_name": getattr(m, "global_name", None),
            "display_name": m.display_name,
        }
        for m in uniq
    ]


async def resolve_user_id(guild_id: int, user_query: str) -> int:
    candidates = await find_members(guild_id, user_query, limit=10)

    if not candidates:
        raise ValueError(
            f"No members found matching '{user_query}'. "
            "Try @mention, numeric ID, exact nickname/display name, or exact username."
        )

    if len(candidates) > 1:
        preview = ", ".join(
            f"{c['display_name']} (id={c['id']})" for c in candidates[:5]
        )
        raise ValueError(
            f"Ambiguous user '{user_query}'. Matches: {preview}. "
            "Please provide a more specific name or an @mention/user ID."
        )

    return int(candidates[0]["id"])


async def resolve_member(guild_id: int, user_query: str) -> discord.Member:
    guild = await _get_guild(guild_id)
    user_id = await resolve_user_id(guild_id, user_query)
    return guild.get_member(user_id) or await guild.fetch_member(user_id)


# ---------------------- MCP TOOLS ----------------------

@mcp.tool()
async def GG_JS_discord_healthcheck() -> str:
    """
    Health check tool that forces a Discord login (if not already logged in)
    and returns the connected bot user.
    Useful to "warm up" the Discord connection so later tool calls are fast.
    """
    await ensure_discord_started()
    return f"ready as {client.user}"


@mcp.tool()
async def GG_JS_discord_send_message(channel_id: int, content: str) -> str:
    """
    Send a message to a Discord channel by channel ID.

    Args:
      channel_id: numeric channel ID
      content: message text (<= ~1900 chars recommended)

    Returns:
      The sent message ID as a string.
    """
    await ensure_discord_started()

    if channel_id <= 0:
        raise ValueError("channel_id must be a positive integer.")
    content = (content or "").strip()
    if not content:
        raise ValueError("content cannot be empty.")
    if len(content) > 1900:
        raise ValueError("content too long; keep under ~1900 characters.")

    channel = client.get_channel(channel_id) or await client.fetch_channel(channel_id)
    msg = await channel.send(content)
    return str(msg.id)


@mcp.tool()
async def GG_JS_discord_send_message_to_channel(
    guild_id: int, channel_ref: str, content: str
) -> str:
    """
    Send a message to a channel by channel reference.

    channel_ref can be:
      - channel ID (e.g., "123456...")
      - channel name (e.g., "#general" or "general")

    Returns:
      The sent message ID.
    """
    channel_id = await resolve_text_channel_id(guild_id, channel_ref)
    return await GG_JS_discord_send_message(channel_id, content)


@mcp.tool()
async def GG_JS_discord_list_text_channels(guild_id: int) -> List[Dict[str, Any]]:
    """
    List all text channels in the given guild.

    Returns:
      [{ "id": <channel_id>, "name": <channel_name> }, ...]
    """
    guild = await _get_guild(guild_id)
    return [{"id": ch.id, "name": ch.name} for ch in guild.text_channels]


@mcp.tool()
async def GG_JS_discord_resolve_text_channel(
    guild_id: int, channel_name: str
) -> Dict[str, Any]:
    """
    Resolve a channel name (like '#ads') into a channel ID.

    Returns:
      { "id": <channel_id>, "name": <provided_name_without_hash> }
    """
    channel_id = await resolve_text_channel_id(guild_id, channel_name)
    return {"id": channel_id, "name": (channel_name or "").lstrip("#").strip()}


@mcp.tool()
async def GG_JS_discord_create_scheduled_event(
    guild_id: int,
    name: str,
    start_time_iso: str,
    end_time_iso: Optional[str] = None,
    description: Optional[str] = None,
    location: Optional[str] = None,
    channel_id: Optional[int] = None,
    entity_type: str = "external",  # "external", "voice", "stage"
    privacy_level: int = 2,
) -> Dict[str, Any]:
    """
    Create a Discord Scheduled Event in a guild.

    Args:
      guild_id: the server/guild ID
      name: event name
      start_time_iso: ISO8601 time with timezone (e.g. 2026-01-23T18:00:00-05:00)
      end_time_iso: optional end time
      description: optional description
      entity_type:
        - "external": requires location
        - "voice": requires channel_id (voice channel)
        - "stage": requires channel_id (stage channel)
      privacy_level: usually 2 (GUILD_ONLY)

    Returns:
      { "id": <event_id>, "name": <event_name>, "url": <event_url> }
    """
    await ensure_discord_started()

    name = (name or "").strip()
    if not name:
        raise ValueError("name is required")

    entity_type_map = {"stage": 1, "voice": 2, "external": 3}
    if entity_type not in entity_type_map:
        raise ValueError("entity_type must be one of: external, voice, stage")

    et = entity_type_map[entity_type]

    if et == 3 and not (location and location.strip()):
        raise ValueError("location is required for external events")

    if et in (1, 2) and not channel_id:
        raise ValueError("channel_id is required for voice/stage events")

    payload: Dict[str, Any] = {
        "name": name,
        "privacy_level": privacy_level,
        "scheduled_start_time": _iso_to_discord(start_time_iso),
        "entity_type": et,
    }

    if end_time_iso:
        payload["scheduled_end_time"] = _iso_to_discord(end_time_iso)

    if description:
        payload["description"] = description.strip()

    if et == 3:
        payload["entity_metadata"] = {"location": location.strip()}
    else:
        payload["channel_id"] = str(channel_id)

    route = discord.http.Route(
        "POST",
        "/guilds/{guild_id}/scheduled-events",
        guild_id=guild_id,
    )
    data = await client.http.request(route, json=payload)

    event_id = data["id"]
    url = f"https://discord.com/events/{guild_id}/{event_id}"
    return {"id": event_id, "name": data["name"], "url": url}


@mcp.tool()
async def GG_JS_get_random_cat_gif() -> str:
    """
    Fetch a random cat GIF URL from TheCatAPI.

    Returns:
      A direct GIF URL as a string.
    """
    headers = {"x-api-key": THECATAPI_KEY} if THECATAPI_KEY else {}
    url = "https://api.thecatapi.com/v1/images/search"
    params = {"limit": 1, "mime_types": "gif"}

    async with httpx.AsyncClient(timeout=10.0) as http:
        r = await http.get(url, params=params, headers=headers)
        r.raise_for_status()
        data = r.json()

    if not data:
        raise RuntimeError("TheCatAPI returned no results.")
    return data[0]["url"]


@mcp.tool()
async def GG_JS_send_cat_gif(channel_id: int) -> str:
    """
    Send a random cat GIF to a Discord channel by channel ID.

    Returns:
      The sent message ID.
    """
    url = await GG_JS_get_random_cat_gif()
    return await GG_JS_discord_send_message(channel_id, url)


@mcp.tool()
async def GG_JS_discord_find_users(
    guild_id: int, user_query: str
) -> List[Dict[str, Any]]:
    """
    Find user candidates in a guild by:
      - @mention (<@id> or <@!id>)
      - numeric user ID
      - username
      - server nickname/display name
      - (best-effort) global_name

    Returns:
      List of candidates: [{id, username, global_name, display_name}, ...]
    """
    if guild_id <= 0:
        raise ValueError("guild_id must be positive")
    return await find_members(guild_id, user_query, limit=10)


@mcp.tool()
async def GG_JS_discord_resolve_user_id(guild_id: int, user_query: str) -> Dict[str, Any]:
    """
    Resolve a user reference to a single user ID.

    user_query can be:
      - @mention
      - numeric ID
      - exact username
      - exact server nickname/display name

    Errors if the match is ambiguous.

    Returns:
      { "user_id": <id> }
    """
    if guild_id <= 0:
        raise ValueError("guild_id must be positive")
    return {"user_id": await resolve_user_id(guild_id, user_query)}


@mcp.tool()
async def GG_JS_discord_find_roles(
    guild_id: int, role_query: str
) -> List[Dict[str, Any]]:
    """
    Find role candidates in a guild by:
      - <@&role_id> mention
      - numeric role ID
      - role name (optionally prefixed with '@')

    Returns:
      List: [{id, name}, ...]
    """
    if guild_id <= 0:
        raise ValueError("guild_id must be positive")
    return await find_roles(guild_id, role_query, limit=10)


@mcp.tool()
async def GG_JS_discord_resolve_role_id(guild_id: int, role_query: str) -> Dict[str, Any]:
    """
    Resolve a role reference to a single role ID.

    role_query can be:
      - <@&id> mention
      - numeric role ID
      - exact role name (or @RoleName)

    Errors if the match is ambiguous.

    Returns:
      { "role_id": <id> }
    """
    if guild_id <= 0:
        raise ValueError("guild_id must be positive")
    return {"role_id": await resolve_role_id(guild_id, role_query)}


@mcp.tool()
async def GG_JS_discord_assign_role(guild_id: int, user_ref: str, role_ref: str) -> str:
    """
    Assign a role to a user, using flexible identifiers.

    Args:
      guild_id: server ID
      user_ref: nickname/username/global_name, @mention, or numeric user ID
      role_ref: role name/@role, <@&role_id>, or numeric role ID

    Returns:
      A human-readable success message.

    Requirements:
      - Bot permission: Manage Roles
      - Bot's top role must be above the role being assigned
    """
    if guild_id <= 0:
        raise ValueError("guild_id must be positive")

    member = await resolve_member(guild_id, user_ref)
    role_id = await resolve_role_id(guild_id, role_ref)
    guild = await _get_guild(guild_id)

    role = guild.get_role(role_id)
    if role is None:
        roles = await guild.fetch_roles()
        role = next((r for r in roles if r.id == role_id), None)
    if role is None:
        raise RuntimeError(f"Role id {role_id} could not be loaded.")

    try:
        await member.add_roles(role)
        return f"Assigned role '{role.name}' to {member.display_name} (id={member.id})."
    except discord.Forbidden:
        raise RuntimeError(
            "Bot lacks permission to assign this role. "
            "Ensure it has Manage Roles and its top role is above the target role."
        )
    except discord.HTTPException as e:
        raise RuntimeError(f"Discord API error: {e}")


@mcp.tool()
async def GG_JS_discord_get_user_info(guild_id: int, user_ref: str) -> Dict[str, Any]:
    """
    Get basic information about a user in a guild.

    user_ref can be nickname/username/global_name, @mention, or numeric ID.

    Returns:
      {
        id, username, global_name, display_name,
        roles: [role_id...],
        role_names: [role_name...]
      }
    """
    if guild_id <= 0:
        raise ValueError("guild_id must be positive")

    member = await resolve_member(guild_id, user_ref)
    return {
        "id": member.id,
        "username": member.name,
        "global_name": getattr(member, "global_name", None),
        "display_name": member.display_name,
        "roles": [r.id for r in member.roles if r.name != "@everyone"],
        "role_names": [r.name for r in member.roles if r.name != "@everyone"],
    }


@mcp.tool()
async def GG_JS_discord_kick_user(guild_id: int, user_ref: str, reason: str = "") -> str:
    """
    Kick a user from a guild.

    user_ref can be nickname/username/global_name, @mention, or numeric ID.

    Returns:
      A human-readable success message.
    """
    if guild_id <= 0:
        raise ValueError("guild_id must be positive")

    member = await resolve_member(guild_id, user_ref)
    await member.kick(reason=(reason or None))
    return f"Kicked {member.display_name} (id={member.id})."


@mcp.tool()
async def GG_JS_discord_ban_user(guild_id: int, user_ref: str, reason: str = "") -> str:
    """
    Ban a user from a guild.

    user_ref can be nickname/username/global_name, @mention, or numeric ID.

    Returns:
      A human-readable success message.
    """
    if guild_id <= 0:
        raise ValueError("guild_id must be positive")

    member = await resolve_member(guild_id, user_ref)
    await member.ban(reason=(reason or None))
    return f"Banned {member.display_name} (id={member.id})."


@mcp.tool()
async def GG_JS_discord_get_recent_messages(
    channel_id: int,
    limit: int = 50,
    before_message_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch recent message history from a channel.

    Requirements:
      - Bot permissions: View Channel, Read Message History

    Args:
      channel_id: channel to read
      limit: number of messages to fetch (1..200)
      before_message_id: optional pagination cursor (fetch messages before this ID)

    Returns:
      list of messages:
      [{
        id, author_name, author_id, content, created_at, jump_url, channel_id
      }, ...]
    """
    await ensure_discord_started()

    if channel_id <= 0:
        raise ValueError("channel_id must be positive")
    if limit < 1 or limit > 200:
        raise ValueError("limit must be between 1 and 200")

    channel = client.get_channel(channel_id) or await client.fetch_channel(channel_id)

    before = None
    if before_message_id:
        try:
            before = discord.Object(id=int(before_message_id))
        except ValueError as e:
            raise ValueError("before_message_id must be a numeric string") from e

    out: List[Dict[str, Any]] = []
    async for msg in channel.history(limit=limit, before=before, oldest_first=False):
        out.append(
            {
                "id": str(msg.id),
                "author_name": msg.author.name,
                "author_id": str(msg.author.id),
                "content": msg.content,
                "created_at": msg.created_at.isoformat(),
                "jump_url": msg.jump_url,
                "channel_id": str(channel_id),
            }
        )

    return out


@mcp.tool()
async def GG_JS_discord_search_recent_messages(
    channel_id: int,
    query: str,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """
    Search recent messages in a channel using a simple substring match.

    This is a lightweight search over the last `limit` messages (not full server search).
    Matching is case-insensitive and checks message.content only.

    Returns:
      A list of messages that matched (same format as GG_JS_discord_get_recent_messages).
    """
    q = (query or "").strip()
    if not q:
        raise ValueError("query cannot be empty")

    msgs = await GG_JS_discord_get_recent_messages(channel_id=channel_id, limit=limit)
    qn = q.lower()

    hits: List[Dict[str, Any]] = []
    for m in msgs:
        if qn in (m.get("content") or "").lower():
            hits.append(m)

    return hits


@mcp.tool()
async def GG_JS_discord_search_recent_messages_in_channel(
    guild_id: int,
    channel_ref: str,
    query: str,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """
    Resolve a channel reference then search recent messages in that channel.

    channel_ref can be:
      - channel ID
      - channel name (#general)

    Returns:
      A list of matching messages (same format as GG_JS_discord_get_recent_messages).
    """
    channel_id = await resolve_text_channel_id(guild_id, channel_ref)
    return await GG_JS_discord_search_recent_messages(
        channel_id=channel_id, query=query, limit=limit
    )


# ---------------------- RUN ----------------------

if __name__ == "__main__":
    mcp.run(transport="streamable-http")