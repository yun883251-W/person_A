"""本机共享画像接口，无多用户鉴权；仅适用于现有localhost部署。"""
from __future__ import annotations

import asyncio
import sqlite3

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, field_validator, model_validator

from knowledge_agent.profile.models import validate_fields
from knowledge_agent.profile.store import ProfileConflictError, load_profile, save_profile


class ProfileWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: StrictBool = True
    fields: dict[str, StrictStr] = Field(default_factory=dict, max_length=6)
    expected_revision: int = Field(ge=0, le=9223372036854775806, strict=True)
    confirmed_fields: list[StrictStr] = Field(default_factory=list, max_length=6)

    @field_validator("fields")
    @classmethod
    def checked_fields(cls, value: dict) -> dict:
        return validate_fields(value)

    @model_validator(mode="after")
    def checked_confirmations(self):
        if len(set(self.confirmed_fields)) != len(self.confirmed_fields) or set(self.confirmed_fields) - set(self.fields):
            raise ValueError("只能重新确认已填写的字段，且不得重复")
        return self


class ProfileClear(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: int = Field(ge=0, le=9223372036854775806, strict=True)

class ProfileProposal(ProfileClear):
    text: StrictStr = Field(min_length=1, max_length=3000)


async def _call(function, *args, **kwargs):
    try:
        return await asyncio.to_thread(function, *args, **kwargs)
    except ProfileConflictError as exc:
        raise HTTPException(409, str(exc)) from None
    except (OSError, sqlite3.Error, ValueError):
        raise HTTPException(503, "画像存储暂时不可用") from None


def register_routes(app: FastAPI) -> None:
    @app.post("/profile/propose", summary="从用户描述提取候选画像，不保存")
    async def propose(body: ProfileProposal):
        from knowledge_agent.profile.extraction import propose_profile
        from knowledge_agent.agent.dialogue.semantic import SemanticProposalError
        from knowledge_agent.models.provider import ChatModelInvocationError
        try:
            return await app.state.runtime.run(propose_profile, body.text, body.expected_revision)
        except ProfileConflictError as exc:
            raise HTTPException(409, str(exc)) from None
        except SemanticProposalError:
            raise HTTPException(502, "模型未返回可核验的画像候选，本次未修改画像。") from None
        except ChatModelInvocationError:
            raise HTTPException(503, "画像提取模型暂时不可用，本次未修改画像。") from None
        except (OSError, sqlite3.Error, ValueError):
            raise HTTPException(503, "画像提取暂时不可用，本次未修改画像。") from None

    @app.get("/profile", summary="查看本机学习画像")
    async def read_profile():
        return await _call(load_profile)

    @app.post("/profile", summary="替换手动学习画像")
    async def write_profile(body: ProfileWrite):
        return await _call(save_profile, body.fields, body.enabled, body.expected_revision,
                           confirmed_fields=body.confirmed_fields)

    @app.post("/profile/clear", summary="清空画像并关闭注入，不删除聊天记录")
    async def clear_profile(body: ProfileClear):
        return await _call(save_profile, {}, False, body.expected_revision)
