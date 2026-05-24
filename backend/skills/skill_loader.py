"""
从仓库 .cursor/skills 读取旅游规划 Skill 正文，供攻略大模型 prompt 拼装。
Skill 源文件与 Cursor Agent 共用，改 prompt 时优先改 .cursor/skills 下 SKILL.md。
"""
from pathlib import Path

# 仓库根目录：backend/skills/skill_loader.py → 上两级
REPO_ROOT = Path(__file__).resolve().parents[2]
CURSOR_SKILLS_DIR = REPO_ROOT / ".cursor" / "skills"

# 攻略生成默认挂载的两份专项 skill
TRAVEL_ITINERARY_SKILL = "travel-itinerary"
TRAVEL_DINING_SKILL = "travel-dining"


def stripYamlFrontmatter(text: str) -> str:
    """去掉 SKILL.md 顶部 YAML frontmatter，只保留 Markdown 正文。"""
    raw = (text or "").lstrip()
    if not raw.startswith("---"):
        return raw
    end = raw.find("\n---", 3)
    if end == -1:
        return raw
    return raw[end + 4 :].lstrip("\n")


def skillMdPath(skillName: str) -> Path:
    """返回指定 skill 目录下 SKILL.md 的路径（skillName 为目录名，如 travel-itinerary）。"""
    return CURSOR_SKILLS_DIR / skillName / "SKILL.md"


def loadSkillMarkdown(skillName: str) -> str:
    """
    读取指定 skill 的 SKILL.md 正文（无 frontmatter）。
    skillName 为目录名，如 travel-itinerary。
    """
    path = skillMdPath(skillName)
    if not path.is_file():
        raise FileNotFoundError(f"未找到旅游规划 skill: {path}")
    return stripYamlFrontmatter(path.read_text(encoding="utf-8")).strip()


def buildCombinedTravelSkillInstruction() -> str:
    """
    合并「行程规划」与「餐饮规划」两份 skill，作为攻略单次 LLM 调用的系统侧规则。
    """
    itineraryBody = loadSkillMarkdown(TRAVEL_ITINERARY_SKILL)
    diningBody = loadSkillMarkdown(TRAVEL_DINING_SKILL)
    combined = (
        "你现在是「旅游攻略编排助手」，须同时严格执行以下两份专项 skill，"
        "输出一篇给用户看的完整攻略正文。\n\n"
        "========== 专项一：行程规划 ==========\n"
        f"{itineraryBody}\n\n"
        "========== 专项二：餐饮规划 ==========\n"
        f"{diningBody}\n\n"
        "========== 合成输出要求 ==========\n"
        "将以上两份 skill 合成一篇攻略，严格按各 skill 规定的 Day 结构与全文顺序输出。\n"
    )
    # 终端可搜 travel_skills=========== 确认本次攻略是否挂载双 skill（重启 main.py 后生效）。
    print(
        "travel_skills===========已加载旅游规划 skill\n"
        f"  - {TRAVEL_ITINERARY_SKILL}: {skillMdPath(TRAVEL_ITINERARY_SKILL)}\n"
        f"  - {TRAVEL_DINING_SKILL}: {skillMdPath(TRAVEL_DINING_SKILL)}\n"
        f"  合并指令字符数: {len(combined)}"
    )
    return combined
