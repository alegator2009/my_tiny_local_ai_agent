from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException

from ..schemas import (
    EvolutionCopyToRootResponse,
    EvolutionDeleteResponse,
    EvolutionEventOut,
    EvolutionGenerationOut,
    EvolutionRunOut,
    EvolutionStart,
)
from ..services.evolution import (
    activate_generation,
    cancel_evolution_run,
    copy_generation_to_root,
    create_evolution_run,
    delete_generation,
    get_evolution_run,
    list_generations,
    list_evolution_events,
    list_evolution_runs,
    run_evolution,
)

router = APIRouter(prefix="/api/evolution", tags=["evolution"])


@router.post("/start", response_model=EvolutionRunOut)
def start_evolution(payload: EvolutionStart, background_tasks: BackgroundTasks):
    run = create_evolution_run(
        prompt=payload.prompt,
        max_generations=payload.max_generations,
        mode=payload.mode,
        stop_on_failure=payload.stop_on_failure,
    )
    background_tasks.add_task(run_evolution, run["id"])
    return run


@router.get("/runs", response_model=list[EvolutionRunOut])
def list_runs_endpoint():
    return list_evolution_runs()


@router.get("/generations", response_model=list[EvolutionGenerationOut])
def list_generations_endpoint():
    return list_generations()


@router.post("/generations/{generation}/activate", response_model=EvolutionGenerationOut)
def activate_generation_endpoint(generation: int):
    try:
        return activate_generation(generation)
    except KeyError:
        raise HTTPException(status_code=404, detail="Generation not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/generations/{generation}/copy-to-root", response_model=EvolutionCopyToRootResponse)
def copy_generation_to_root_endpoint(generation: int):
    try:
        return copy_generation_to_root(generation)
    except KeyError:
        raise HTTPException(status_code=404, detail="Generation not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/generations/{generation}", response_model=EvolutionDeleteResponse)
def delete_generation_endpoint(generation: int, force: bool = False):
    try:
        return delete_generation(generation, force=force)
    except KeyError:
        raise HTTPException(status_code=404, detail="Generation not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/runs/{run_id}", response_model=EvolutionRunOut)
def get_run_endpoint(run_id: str):
    try:
        return get_evolution_run(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Evolution run not found")


@router.get("/runs/{run_id}/events", response_model=list[EvolutionEventOut])
def get_events_endpoint(run_id: str):
    try:
        return list_evolution_events(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Evolution run not found")


@router.post("/runs/{run_id}/cancel", response_model=EvolutionRunOut)
def cancel_run_endpoint(run_id: str):
    try:
        return cancel_evolution_run(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Evolution run not found")
