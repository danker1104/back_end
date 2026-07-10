# 대한민국 생활 법률 AI 챗봇

이 프로젝트는 PRD에 따라 EXAONE-4.0-1.2B 모델과 RAG 파이프라인의 1단계를 구성합니다.

## 1단계 범위

- 프로젝트 폴더 구조 생성
- 의존성 정의
- 기본 설정 파일 작성
- Hugging Face에서 EXAONE-4.0-1.2B 모델을 다운로드하고 로드하는 기본 코드 추가

## 폴더 구조

```text
project/
├── config/
├── data/
├── embeddings/
├── inference/
├── model/
├── preprocessing/
├── rag/
├── utils/
├── vector_db/
├── output/
├── build_vector_db.py
├── chat.py
├── requirements.txt
└── README.md
```

## 실행 준비

1. Python 3.10 이상을 준비합니다.
2. 의존성을 설치합니다.

```bash
pip install -r requirements.txt
```

3. Hugging Face 인증이 필요할 경우 환경 변수로 토큰을 설정합니다.

```bash
set HF_TOKEN=your_huggingface_token
```

## 기본 모델 로딩 예시

모델 로딩 코드는 [config.py](config.py)와 [model/load_model.py](model/load_model.py)에서 관리합니다.

## 참고

- 데이터 전처리, JSON 분석, FAISS, RAG, Embedding 구현은 1단계에서 포함하지 않습니다.
- 이후 단계에서 순차적으로 확장합니다.
