"""Model management tools: list / register / optimize / deploy."""

from __future__ import annotations

from typing import Any, Dict, List

from thor_mcp.tools import ToolError

SPECS: List[Dict[str, Any]] = [
    {
        "name": "models_list",
        "description": "List registered models",
        "properties": {
            "architecture": {"type": "string"},
            "optimized": {"type": "boolean"},
        },
        "required": [],
    },
    {
        "name": "models_register",
        "description": "Register a model in the Thor model registry",
        "properties": {
            "model_id": {"type": "string"},
            "name": {"type": "string"},
            "architecture": {"type": "string"},
            "parameters": {"type": "integer"},
            "source": {"type": "string", "enum": ["huggingface", "ultralytics", "custom", "timm"]},
            "license": {"type": "string"},
            "metadata": {"type": "object"},
        },
        "required": ["model_id"],
    },
    {
        "name": "models_optimize",
        "description": "Create an optimization profile for Thor deployment",
        "properties": {
            "model_id": {"type": "string"},
            "optimization_type": {"type": "string", "enum": ["tensorrt", "quantization", "pruning", "distillation"]},
            "precision": {"type": "string", "enum": ["fp32", "fp16", "int8", "int4", "fp8"]},
            "target_latency_ms": {"type": "number"},
            "target_throughput": {"type": "number"},
            "target_memory_mb": {"type": "number"},
            "enable_sparsity": {"type": "boolean", "default": False},
            "execute": {"type": "boolean", "default": False, "description": "Execute the build (TensorRT toolchain on Thor, or torch + model_path for int8 quantization)"},
            "model_path": {"type": "string", "description": "Local path to a torch model save, required to execute int8 quantization"},
            "output_path": {"type": "string", "description": "Engine .plan output path (tensorrt execute); defaults to the shared engine cache so benchmark_run --backend tensorrt can auto-discover it"},
        },
        "required": ["model_id", "optimization_type"],
    },
    {
        "name": "models_deploy",
        "description": "Create a deployment descriptor for an optimized model",
        "properties": {
            "model_id": {"type": "string"},
            "precision": {"type": "string"},
            "profile_id": {"type": "string", "description": "Optimization profile to deploy"},
        },
        "required": ["model_id"],
    },
]

HANDLERS: Dict[str, Any] = {}


async def models_list(args: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    models = ctx.registry.list(
        architecture=args.get("architecture"),
        optimized=args.get("optimized"),
    )
    return {"count": len(models), "models": models}


async def models_register(args: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    model_id = args.get("model_id")
    if not model_id:
        raise ToolError("model_id is required")
    entry = ctx.registry.register(
        model_id=model_id,
        name=args.get("name"),
        architecture=args.get("architecture"),
        parameters=args.get("parameters"),
        source=args.get("source"),
        license=args.get("license"),
        metadata=args.get("metadata"),
    )
    return {"status": "success", "model": entry}


async def models_optimize(args: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    from thor_models.optimize.profiles import OptimizeError, optimize_model

    model_id = args.get("model_id")
    if not model_id:
        raise ToolError("model_id is required")
    try:
        return optimize_model(
            model_id=model_id,
            optimization_type=args.get("optimization_type", "tensorrt"),
            precision=args.get("precision", "fp16"),
            target_latency_ms=args.get("target_latency_ms"),
            target_throughput=args.get("target_throughput"),
            target_memory_mb=args.get("target_memory_mb"),
            enable_sparsity=args.get("enable_sparsity", False),
            execute=args.get("execute", False),
            model_path=args.get("model_path"),
            output_path=args.get("output_path"),
            cache_dir=ctx.config.models.cache_dir,
        )
    except OptimizeError as exc:
        raise ToolError(str(exc)) from exc


async def models_deploy(args: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    model_id = args.get("model_id")
    if not model_id:
        raise ToolError("model_id is required")
    entry = ctx.registry.get(model_id)
    if entry is None:
        raise ToolError(f"model not registered: {model_id}")
    entry.setdefault("metadata", {})["optimized"] = True
    entry["metadata"]["deployment"] = {
        "precision": args.get("precision", "fp16"),
        "profile_id": args.get("profile_id"),
        "status": "staged",
    }
    return {
        "status": "success",
        "model_id": model_id,
        "deployment": entry["metadata"]["deployment"],
        "note": "Deployment descriptor created; engine build requires the optimization toolchain.",
    }


HANDLERS.update({
    "models_list": models_list,
    "models_register": models_register,
    "models_optimize": models_optimize,
    "models_deploy": models_deploy,
})
