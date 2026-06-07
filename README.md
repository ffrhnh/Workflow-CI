# Workflow-CI — Titanic MLflow Project

Repository ini berisi workflow CI untuk re-training model Titanic secara otomatis menggunakan **MLflow Project** dan **GitHub Actions**.

---

## 📁 Struktur Repository

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci.yml                  ← GitHub Actions CI
└── MLProject/
    ├── modelling.py                ← Script training
    ├── conda.yaml                  ← Environment dependencies
    ├── MLProject                   ← Konfigurasi MLflow Project
    ├── Tautan_DockerHub.txt        ← Link Docker Hub
    └── titanic_preprocessing/
        ├── titanic_train.csv
        └── titanic_test.csv
```

---

## 🔄 Trigger CI

Workflow otomatis berjalan ketika:
- **Push** ke branch `main` yang mengubah file di folder `MLProject/`
- **Pull Request** ke `main`
- **Manual** melalui tab Actions → Run workflow

---

## ⚙️ Yang Dilakukan Workflow

| Job | Langkah | Level |
|---|---|---|
| **train** | Checkout → Install deps → `mlflow run` → Upload artefak → Commit ke repo | Skilled |
| **docker** | Login Docker Hub → `mlflow models build-docker` → Push image | Advance |

---

## 🔐 GitHub Secrets yang Dibutuhkan

Tambahkan di **Settings → Secrets and variables → Actions**:

| Secret | Nilai |
|---|---|
| `DOCKER_USERNAME` | Username Docker Hub Anda |
| `DOCKER_TOKEN` | Access Token Docker Hub |

---

## 🐳 Docker Hub

Image tersedia di:
```
docker pull username_dockerhub_anda/titanic-mlflow:latest
```
