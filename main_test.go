package main

import "testing"

func TestVectorise(t *testing.T){
	documents := []string{
		"passage: Hello, World!",
		"passage: Mickey Mouse was on Disney",
		"passage: fastembed-go is licensed under MIT",
	}
	embeddings, err := Vectorise(documents)
	if err != nil {
		t.Error("embedding failed")
	}
	if len(embeddings) != 3 || len(embeddings[0]) != 768{
		t.Error(`Embeddings not generating in expected shape`)
	}
}

func TestSearchIndex(t *testing.T){
	documents := []string{
	"Hello",
	"passage: This is an example passage.",
	"passage: Mickey Mouse was on Disney",
	"fastembed-go is licensed under MIT",
	}

	embeddings, err := Vectorise(documents)
	if err != nil {
		t.Error(err)
	}

	query := []string{
		"query: Who was on disney?",
	}

	f := FlatIndex{Vectors: embeddings, Text: documents}

	v_match, err := f.SearchIndex(query)
	if err != nil {
		t.Error(err)
	}

	want := "passage: Mickey Mouse was on Disney"
	got := v_match.Text

	if got != want{
		t.Errorf(`Want: %q, Got: %v`,want, got)
	}
}