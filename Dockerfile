FROM python:3.13-alpine

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install build dependencies (incl. C++ compiler for matplotlib)
RUN apk add --no-cache gcc g++ musl-dev libffi-dev

COPY ./pyproject.toml /app
COPY ./.python-version /app
COPY ./uv.lock /app
RUN uv sync --locked

COPY ./src /app/src
COPY ./pages /app/pages
COPY ./sim /app/sim
COPY ./main.py /app/main.py

EXPOSE 80

CMD ["uv", "run", "streamlit", "run", "main.py", "--server.port=80", "--server.address=0.0.0.0"]
