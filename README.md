# ai-defra-search-agent

[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=DEFRA_ai-defra-search-frontend&metric=security_rating)](https://sonarcloud.io/summary/new_code?id=DEFRA_ai-defra-search-agent)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=DEFRA_ai-defra-search-frontend&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=DEFRA_ai-defra-search-agent)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=DEFRA_ai-defra-search-frontend&metric=coverage)](https://sonarcloud.io/summary/new_code?id=DEFRA_ai-defra-search-agent)

Agent service for the AI DEFRA Search application. This service provides the AI Assistant backend, handling chat interactions and knowledge retrieval.

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Environment Variables](#environment-variables)
- [Running the Application](#running-the-application)
  - [Building the Docker Image](#building-the-docker-image)
  - [Starting the Docker Container](#starting-the-docker-container)
  - [Debugging](#debugging)
- [Development Tools](#development-tools)
  - [Linting and Formatting](#linting-and-formatting)
  - [VS Code Configuration](#vs-code-configuration)
  - [IntelliJ Configuration](#intellij-configuration)
- [Tests](#tests)
- [API Endpoints](#api-endpoints)
- [Custom CloudWatch Metrics](#custom-cloudwatch-metrics)
- [Licence](#licence)
  - [About the licence](#about-the-licence)

## Prerequisites

- Docker
- Docker Compose
- Python >= 3.13.7
- pipx

## Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/DEFRA/ai-defra-search-agent.git
cd ai-defra-search-agent
pipx install uv
uv sync
```

This service uses the [FastAPI](https://fastapi.tiangolo.com/) Python API framework and [uv](https://github.com/astral-sh/uv) for dependency management.

**Apple Silicon Users:**
If using Apple Silicon, you will need to install the `arm64` version of Python 3.13.7:

```bash
brew install python@3.13
uv venv --python /opt/homebrew/bin/python3.13
```

## Environment Variables

Create a `.env` file in the root of the project from the example template:

```bash
cp .env.example .env
```

The following environment variables can be configured for the application:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AWS_REGION` | Yes | `eu-central-1` | The AWS region to use for AWS services |
| `AWS_DEFAULT_REGION` | Yes | `eu-central-1` | The default AWS region (should match AWS_REGION) |
| `AWS_ACCESS_KEY_ID` | Yes | `test` | AWS access key ID (use `test` for local development with Localstack) |
| `AWS_SECRET_ACCESS_KEY` | Yes | `test` | AWS secret access key (use `test` for local development with Localstack) |
| `AWS_EMF_ENVIRONMENT` | Yes | `local` | AWS Embedded Metrics environment setting |
| `AWS_EMF_AGENT_ENDPOINT` | Yes | `tcp://127.0.0.1:25888` | CloudWatch agent endpoint for metrics |
| `AWS_EMF_LOG_GROUP_NAME` | Yes | `log-group-name` | CloudWatch log group name |
| `AWS_EMF_LOG_STREAM_NAME` | Yes | `log-stream-name` | CloudWatch log stream name |
| `AWS_EMF_NAMESPACE` | Yes | `namespace` | CloudWatch metrics namespace |
| `AWS_EMF_SERVICE_NAME` | Yes | `service-name` | Service name for CloudWatch metrics |
| `AWS_EMF_SERVICE_TYPE` | Yes | `python-backend-service` | Service type identifier |
| `AWS_BEARER_TOKEN_BEDROCK` | No | N/A | Bearer token for AWS Bedrock authentication |
| `AWS_BEDROCK_DEFAULT_GENERATION_MODEL` | Yes | N/A | Default AI model to use for generation |
| `AWS_BEDROCK_AVAILABLE_GENERATION_MODELS` | Yes | N/A | JSON array of available AI models for generation |

In CDP, environment variables and secrets need to be set using CDP conventions:
- [CDP App Config](https://github.com/DEFRA/cdp-documentation/blob/main/how-to/config.md)
- [CDP Secrets](https://github.com/DEFRA/cdp-documentation/blob/main/how-to/secrets.md)

## Running the Application

### Building the Docker Image

Container images are built using Docker Compose. First, build the Docker image:

```bash
docker compose build
```

### Starting the Docker Container

After building the image, run the service locally in a container alongside MongoDB:

```bash
docker compose --profile service up
```

Use the `-d` flag at the end of the above command to run in detached mode (e.g., if you wish to view logs in another application such as Docker Desktop):

```bash
docker compose --profile service up -d
```

The application will be available at `http://localhost:8086`.

If you want to enable hot-reloading, you can press the `w` key once the compose project is running to enable `watch` mode.

To stop the containers:

```bash
docker compose down
```

## Development Tools

### Debugging

This project is configured for debugging with `debugpy`. Follow the instructions below for your IDE to set up the debugging environment.

**Visual Studio Code:**

1. Install the [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) and [Python Debugger](https://marketplace.visualstudio.com/items?itemName=ms-python.debugpy) extensions for Visual Studio Code.
2. Open the command palette (Ctrl+Shift+P) and select "Debug: Add Configuration".
3. In the dropdown, select "Python Debugger" -> "Python: Remote Attach".
4. Enter the following configuration:
   - host => localhost
   - port => 5679
   - localRoot => ${workspaceFolder}
   - remoteRoot => /home/nonroot

Start the service in debug mode by running:

```bash
docker compose -f compose.yml -f compose.debug.yml up --build
```

You should now be able to attach the debugger to the running service.

### Linting and Formatting

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting Python code. Ruff is configured in the `.ruff.toml` file.

To run Ruff from the command line:

```bash
# Run linting with auto-fix
uv run ruff check --fix

# Run formatting
uv run ruff format

# Or use the taskipy shortcut
uv run task lint
```

### VS Code Configuration

For the best development experience, configure VS Code to use Ruff:

1. Install the [Ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) for VS Code
2. Configure your VS Code settings (`.vscode/settings.json`):

```json
{
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.fixAll.ruff": "explicit",
        "source.organizeImports.ruff": "explicit"
    },
    "ruff.lint.run": "onSave",
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.fixAll.ruff": "explicit",
            "source.organizeImports.ruff": "explicit"
        }
    }
}
```

This configuration will:
- Format your code with Ruff when you save a file
- Fix linting issues automatically when possible
- Organize imports according to isort rules

### IntelliJ Configuration

**Configure Python Interpreter:**

1. Open **File → Project Structure** 
2. Navigate to **Platform settings → SDKs**
3. Click **Add SDK → Python SDK from disk → Existing Environment**
4. Set type to uv
5. Select Uv env to use your local venv:
   ```
   [project-root]/ai-defra-search-agent/.venv/bin/python
   ```
6. Click **OK** to apply

**Configure Test Environment:**

1. Go to **Run → Edit Configurations**
2. Select or create your test configuration
3. Add required environment variables:
   - Click the **Environment variables** field
   - Click the folder icon to add variables
   - Add all necessary variables (see environment configuration section)
4. Click **OK** to save

## Tests

### Running Tests

Run the tests with:

```bash
uv run task docker-test
```

This command will:
1. Stop any running containers
2. Build the service
3. Run the test suite using pytest
4. Generate coverage reports in the `./coverage` directory

Testing follows the [FastAPI documented approach](https://fastapi.tiangolo.com/tutorial/testing/), using pytest and Starlette.

To run tests locally without Docker:

```bash
uv run task test
```

## API Endpoints

| Endpoint                    | Description                           |
| :-------------------------- | :------------------------------------ |
| `GET: /docs`                | Automatic API Swagger documentation   |
| `GET: /health`              | Health check endpoint                 |
| `POST: /chat`               | Chat interaction with AI assistant    |

## Custom CloudWatch Metrics

This service uses the [AWS Embedded Metrics library](https://github.com/awslabs/aws-embedded-metrics-python) for publishing custom metrics to CloudWatch. An example implementation can be found in `app/common/metrics.py`.

The environment variable `AWS_EMF_ENVIRONMENT=local` is set in the app config to enable integration with the CloudWatch agent configured in CDP. The following CDP environment variables are used:
- `AWS_EMF_AGENT_ENDPOINT`
- `AWS_EMF_LOG_GROUP_NAME`
- `AWS_EMF_LOG_STREAM_NAME`
- `AWS_EMF_NAMESPACE`
- `AWS_EMF_SERVICE_NAME`

## Licence

THIS INFORMATION IS LICENSED UNDER THE CONDITIONS OF THE OPEN GOVERNMENT LICENCE found at:

<http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3>

The following attribution statement MUST be cited in your products and applications when using this information.

> Contains public sector information licensed under the Open Government license v3

### About the licence

The Open Government Licence (OGL) was developed by the Controller of Her Majesty's Stationery Office (HMSO) to enable
information providers in the public sector to license the use and re-use of their information under a common open
licence.

It is designed to encourage use and re-use of information freely and flexibly, with only a few conditions.
