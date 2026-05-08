# Configuration

Example configuration files live in `configs/`.

Do not commit local credentials. Use either environment variables or ignored local YAML files such as:

```bash
cp configs/supramas.example.yaml configs/supramas.local.yaml
```

Important runtime services:

- OpenAI-compatible LLM endpoint
- Embedding endpoint
- Milvus vector database
- Optional GROBID service for PDF parsing
- Optional Volcengine translation credentials

