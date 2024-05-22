from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware

from database import SessionLocal, QA

# Inicializar FastAPI
app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo y tokenizador preentrenados
qa_pipeline = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad', tokenizer='distilbert-base-uncased-distilled-squad')

# Dependencia para obtener la sesión de la base de datos
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Definir el esquema de la solicitud
class Query(BaseModel):
    question: str
    context: str

@app.post('/ask')
async def ask_question(query: Query, db: Session = Depends(get_db)):
    try:
        result = qa_pipeline(question=query.question, context=query.context)
        answer = result['answer']
        
        # Almacenar la pregunta y respuesta en la base de datos
        qa = QA(question=query.question, answer=answer)
        db.add(qa)
        db.commit()
        db.refresh(qa)
        
        return {"answer": answer}
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.get('/questions')
async def get_questions(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    try:
        questions = db.query(QA).offset(skip).limit(limit).all()
        return questions
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# Iniciar la API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
