from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import Optional
from typing import UUID

class BenchmarkResultSchema(BaseModel):
    model: str
    concurrency: int
    duration: float
    successful_requests: int
    input_tokens: int
    output_tokens: int
    total_input_tokens: Optional[int] = 0
    total_output_tokens: Optional[int] = 0
    request_throughput: Optional[float] = None
    input_throughput: Optional[float] = None
    output_throughput: Optional[float] = None
    output_throughput_per_user: Optional[float] = None
    p25_throughput: Optional[float] = None
    p75_throughput: Optional[float] = None
    p95_throughput: Optional[float] = None    
    p99_throughput: Optional[float] = None
    min_throughput: Optional[float] = None
    max_throughput: Optional[float] = None
    mean_ttft_ms: Optional[float] = None
    median_ttft_ms: Optional[float] = None
    p25_ttft_ms: Optional[float] = None
    p75_ttft_ms: Optional[float] = None
    p95_ttft_ms: Optional[float] = None
    p99_ttft_ms: Optional[float] = None
    min_ttft_ms: Optional[float] = None
    max_ttft_ms: Optional[float] = None
    mean_tpot_ms: Optional[float] = None
    median_tpot_ms: Optional[float] = None
    p25_tpot_ms: Optional[float] = None
    p75_tpot_ms: Optional[float] = None
    p95_tpot_ms: Optional[float] = None
    p99_tpot_ms: Optional[float] = None
    min_tpot_ms: Optional[float] = None
    max_tpot_ms: Optional[float] = None
    mean_itl_ms: Optional[float] = None
    median_itl_ms: Optional[float] = None
    p25_itl_ms: Optional[float] = None
    p75_itl_ms: Optional[float] = None
    p95_itl_ms: Optional[float] = None
    p99_itl_ms: Optional[float] = None
    min_itl_ms: Optional[float] = None
    max_itl_ms: Optional[float] = None
    mean_e2el_ms: Optional[float] = None
    median_e2el_ms: Optional[float] = None
    p25_e2el_ms: Optional[float] = None
    p75_e2el_ms: Optional[float] = None
    p95_e2el_ms: Optional[float] = None
    p99_e2el_ms: Optional[float] = None
    min_e2el_ms: Optional[float] = None
    max_e2el_ms: Optional[float] = None
    error_messages: Optional[list] = []


class BenchmarkRequestMetrics(BaseModel):
    benchmark_id: UUID | None = Field(None, alias="benchmark_id")
    dataset_id: UUID | None = Field(None, alias="dataset_id")
    latency: float | None = Field(None, alias="end_to_end_latency_s")
    success: bool | None = Field(True, alias="success")
    error: str | None = Field(None, alias="error_msg")
    prompt_len: int | None = Field(None, alias="number_input_tokens")
    output_len: int | None = Field(None, alias="number_output_tokens")
    req_output_throughput: float | None = Field(None, alias="request_output_throughput_token_per_s")
    ttft: float | None = Field(None, alias="ttft_s")
    tpot: float | None = Field(None, alias="tpot_s")
    itl: list | None = Field(None, alias="inter_token_latency_s")
    
    model_config = ConfigDict(extra="allow")
    
    @model_validator(mode="before")
    @classmethod
    def handle_multiple_aliases_for_error(cls, values):
        if "error_message" in values:
            values["error_msg"] = values.pop("error_message")
        if "error_code" in values and values["error_code"] is not None:
            values["success"] = False
        return values

