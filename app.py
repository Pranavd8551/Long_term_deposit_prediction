
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run

from typing import Optional

from long_term_deposit_prediction.constants import APP_HOST, APP_PORT
from long_term_deposit_prediction.pipline.prediction_pipeline import DepositData, DepositClassifier
from long_term_deposit_prediction.pipline.training_pipeline import TrainPipeline

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory='templates')

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.age: Optional[str] = None
        self.job: Optional[str] = None
        self.marital: Optional[str] = None
        self.education: Optional[str] = None
        self.contact: Optional[str] = None
        self.month: Optional[str] = None
        self.duration: Optional[str] = None
        self.campaign: Optional[str] = None
        self.poutcome: Optional[str] = None
        self.emp_var_rate: Optional[str] = None
        self.cons_conf_idx: Optional[str] = None
        

    async def get_usvisa_data(self):
        form = await self.request.form()
        self.age = form.get("age")
        self.job = form.get("job")
        self.marital = form.get("marital")
        self.education = form.get("education")
        self.contact = form.get("contact")
        self.month = form.get("month")
        self.duration = form.get("duration")
        self.campaign = form.get("campaign")
        self.poutcome = form.get("poutcome")
        self.emp_var_rate = form.get("emp_var_rate")
        self.cons_conf_idx = form.get("cons_conf_idx")

@app.get("/", tags=["authentication"])
async def index(request: Request):

    return templates.TemplateResponse(
            "long_term_deposit_prediction.html",{"request": request, "context": "Rendering"})


@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()

        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/")
async def predictRouteClient(request: Request):
    try:
        form = DataForm(request)
        await form.get_deposit_data()
        
        deposit_data = DepositData(
                                age= form.age,
                                job = form.job,
                                marital = form.marital,
                                education = form.education,
                                contact= form.contact,
                                month= form.month,
                                duration = form.duration,
                                campaign= form.campaign,
                                poutcome= form.poutcome,
                                emp_var_rate= form.emp_var_rate,
                                cons_conf_idx= form.cons_conf_idx,
                                )
        
        deposit_df = deposit_data.get_deposit_input_data_frame()

        model_predictor = DepositClassifier()

        value = model_predictor.predict(dataframe=deposit_df)[0]

        status = None
        if value == 1:
            status = "Long Term Deposit Subscribed"
        else:
            status = "Long Term Deposit not Subscribed"

        return templates.TemplateResponse(
            "long_term_deposit_prediction.html",
            {"request": request, "context": status},
        )
        
    except Exception as e:
        return {"status": False, "error": f"{e}"}


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)