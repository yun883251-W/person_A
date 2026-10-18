"""网页静态入口、健康状态和 Skill 目录。"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from knowledge_agent.skills.loader import discover_skills
from ..schemas import HealthResponse, SkillListResponse, SkillResponse


def register_routes(application: FastAPI, *, web_directory: Path) -> None:
    """注册无状态系统端点与本地网页静态资源。"""
    # 只公开网页资源目录；仓库、配置、文档和数据库不通过静态路由暴露。
    application.mount(
        "/assets",
        StaticFiles(directory=web_directory / "assets", check_dir=False),
        name="assets",
    )

    @application.get("/", response_class=FileResponse, include_in_schema=False)
    async def web_index() -> FileResponse:
        page = web_directory / "index.html"
        if not page.is_file():
            # 网页缺失时保留普通 404，不使用会话专属错误码或覆盖 API 路由。
            raise StarletteHTTPException(status_code=status.HTTP_404_NOT_FOUND)
        return FileResponse(page, media_type="text/html")

    @application.get(
        "/skills",
        response_model=SkillListResponse,
        summary="列出可手动选择的技能",
    )
    async def list_skills() -> SkillListResponse:
        return SkillListResponse(
            items=[
                SkillResponse(
                    skill_id=skill.skill_id,
                    name=skill.name,
                    description=skill.description,
                )
                for skill in discover_skills()
            ]
        )

    @application.get(
        "/health",
        response_model=HealthResponse,
        summary="检查服务状态",
        description="仅返回服务名称、版本、忙碌状态和存活状态，不调用模型或知识库。",
    )
    async def health() -> HealthResponse:
        return HealthResponse(
            busy=application.state.runtime.busy,
            service=application.title,
            version=application.version,
        )
