# vector-db

A simple and efficient vector database written in Go.

## Features

- Fast vector indexing and search
- Embedding generation using fastembed
- Flat-index search with cosine similarity

## Getting Started

### Prerequisites

- Go 1.20+
- ONNX Runtime (see below)

### Installation

Clone the repository:

```sh
git clone https://github.com/sharath-09/vector-db.git
cd vector-db
```

Install dependencies:

```sh
go run main.go
```

Embeddings generated using `BGEBaseENV15`.

### Setting up ONNX for Embedding Generation

Download a ONNX runtime file from [onnxruntime_go_examples](https://github.com/yalue/onnxruntime_go_examples/tree/master/third_party), and set the `ONNX_PATH` environment variable to the `.dylib` file for mac, or `.so` for linux


## Testing

Run all tests:

```sh
go test ./...
```

## References

- [fastembed-go](https://github.com/Anush008/fastembed-go/tree/main)
- [vector-db distanceMeasure.go](https://github.com/tobias-mayer/vector-db/blob/master/pkg/index/distanceMeasure.go#L15)

## License

MIT License. See [LICENSE](LICENSE) for