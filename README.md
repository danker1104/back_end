# 다른 컴퓨터에서 프로젝트 실행하기 (수신자용)

아래 절차만 따라 하면 클론한 저장소를 실행 가능한 상태로 복원할 수 있습니다.

## 1. 저장소 클론

```powershell
git clone https://github.com/danker1104/-_-.git
cd -_-
```

## 2. Python 가상환경 생성 및 활성화

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

## 3. 패키지 설치

```powershell
pip install -r requirements.txt
```

## 4. Google Drive에서 대용량 폴더 다운로드 및 복원

이 저장소에는 대용량 파일이 제외되어 있으므로, 공유받은 압축 파일을 내려받아 프로젝트 루트에 복원해야 합니다.

필수 복원 대상:
- data/
- model/
- embeddings/
- vector_db/

복원 후 최종 경로 예시:
- data/TS_01. 민사법_001. 판결문/
- data/TS_01. 민사법_002. 법령/
- model/...
- embeddings/...
- vector_db/...

## 5. Hugging Face 인증(필요한 경우)

모델 접근 권한이 필요한 경우 로그인합니다.

```powershell
huggingface-cli login
```

## 6. 벡터 DB 재생성(필요한 경우)

vector_db 복원을 하지 않았거나 비어 있으면 아래 명령으로 생성합니다.

```powershell
python build_vector_db.py
```

## 7. 챗 실행

```powershell
python chat.py
```

## 8. 동작 확인

아래 흐름이 정상 동작하면 복원이 완료된 상태입니다.
- 질문 입력
- RAG 검색(Top-K)
- EXAONE 답변 출력
