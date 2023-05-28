import uvicorn
from fastapi import FastAPI

from model import Model
from request_body import TrackRequestBody, ModeResponse

app = FastAPI()


if __name__ == "__main__":

    model_basic = Model('basic')
    model_advanced = Model('advanced')

    @app.post("/basic/mode")
    async def basic_mode(track: TrackRequestBody):
        track_df = track.to_df()
        mode = model_basic.predict(track_df)
        return ModeResponse(mode=mode)

    @app.post("/advanced/mode")
    async def advanced_mode(track: TrackRequestBody):
        track_df = track.to_df()
        mode = model_advanced.predict(track_df)
        return ModeResponse(mode=mode)

    uvicorn.run(app, host='0.0.0.0', port=8000)
