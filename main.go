package main

import (
	"errors"
	"fmt"
	"log"
	"os"

	"github.com/anush008/fastembed-go"
	"github.com/chewxy/math32"
)

type FlatIndex struct {
	Vectors [][]float32
	Text    []string
}

type GraphNode struct {
	Embedding []float32
	Text      string
}

func cosineDistance(v1 []float32, v2 []float32) float32 {
	//reference implemenation: https://github.com/tobias-mayer/vector-db/blob/f4bae4b955a151ac4a524a11b5d08993e4a32d61/pkg/index/distanceMeasure.go#L15
	if len(v1) != len(v2) || len(v1) == 0 {
		return 0
	}

	var dotProduct float32 = 0.0
	var magA float32 = 0.0
	var magB float32 = 0.0

	for i := 0; i < len(v1); i++ {
		dotProduct += v1[i] * v2[i]
		magA += v1[i] * v2[i]
		magB += v2[i] * v2[i]
	}

	magA = math32.Sqrt(magA)
	magB = math32.Sqrt(magB)

	if magA == 0 || magB == 0 {
		return 0.0
	}

	return -dotProduct / (magA * magB)
}

func (f FlatIndex) SearchIndex(query []string) (GraphNode, error) {
	//embed query
	queryEmbedding, err := Vectorise(query)
	if err != nil {
		log.Fatal(err)
	}
	if len(queryEmbedding[0]) != len(f.Vectors[0]) {
		return GraphNode{}, errors.New("mismatch vector shape")
	}
	var closestVector GraphNode
	var shortestDistance float32 = math32.MaxFloat32
	var idx2 = 0
	for idx := range f.Vectors {
		if idx2 >= len(queryEmbedding) {
			idx2 = 0
		}
		distance := cosineDistance(queryEmbedding[idx2], f.Vectors[idx])
		if distance < shortestDistance {
			shortestDistance = distance
			closestVector = GraphNode{Embedding: f.Vectors[idx], Text: f.Text[idx]}
		}
		idx2 += 1
	}
	return closestVector, nil
}

func Vectorise(documents []string) ([][]float32, error) {
	options := fastembed.InitOptions{
		Model:     fastembed.BGEBaseENV15,
		CacheDir:  "model_cache",
		MaxLength: 200,
	}

	model, err := fastembed.NewFlagEmbedding(&options)
	if err != nil {
		panic(err)
	}

	defer model.Destroy()

	// Generate embeddings with a batch-size of 25, defaults to 256
	embeddings, err := model.Embed(documents, 25) //  -> Embeddings length: 4
	if err != nil {
		panic(err)
	}
	return embeddings, nil
}

func main() {

	onnxPath := os.Getenv("ONNX_PATH")
	if onnxPath == ""{
		//Try setting ONNX_PATH
		os.Setenv("ONNX_PATH", "./onnx_dylib/onnxruntime_arm64.dylib")
	}

	documents := []string{
		"Hello",
		"passage: This is an example passage.",
		"passage: Mickey Mouse was on Disney",
		"fastembed-go is licensed under MIT",
	}

	embeddings, err := Vectorise(documents)
	if err != nil {
		log.Fatal(err)
	}

	query := []string{
		"query: Who was on disney?",
	}

	f := FlatIndex{Vectors: embeddings, Text: documents}

	v_match, err := f.SearchIndex(query)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(v_match.Text)
}
