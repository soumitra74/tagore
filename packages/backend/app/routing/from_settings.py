from app.config import Settings
from app.routing.client import OpenAICompatibleClient
from app.routing.router import DynamicRouter, StaticTableStrategy, TaskType, Tier


def tiers_from_settings(settings: Settings) -> list[Tier]:
    return [Tier(**t.model_dump()) for t in settings.router_tiers]


def static_strategy_from_settings(settings: Settings) -> StaticTableStrategy:
    valid = {t.value for t in TaskType}
    unknown = sorted(set(settings.router_task_tiers) - valid)
    if unknown:
        raise ValueError(f"router_task_tiers has unknown task types: {unknown}")
    table = {TaskType(task): tier for task, tier in settings.router_task_tiers.items()}
    return StaticTableStrategy(table, default=settings.router_fallback_tier)


def create_router(settings: Settings):
    """Returns None when no tiers are configured (routing off)."""
    if not settings.router_tiers:
        return None
    return DynamicRouter(
        tiers_from_settings(settings),
        static_strategy_from_settings(settings),
        OpenAICompatibleClient(),
        fallback_tier=settings.router_fallback_tier,
    )


def judge_identity(settings: Settings) -> tuple[str, str]:
    """(provider, model) recorded on an eval cycle for its judge."""
    if not settings.router_tiers:
        return settings.eval_judge_provider, settings.eval_judge_model
    tier_name = settings.router_task_tiers.get(TaskType.JUDGE.value, settings.router_fallback_tier)
    tier = next(t for t in settings.router_tiers if t.name == tier_name)
    return "router", f"{tier.name}:{tier.model}"
