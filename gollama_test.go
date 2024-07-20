package gollama

import (
	"errors"
	"fmt"
	"log"
	"strconv"
	"strings"
	"testing"
)

var docs = []string{
	`Michael Burnham is the main character on the Star Trek series, Discovery.  
	She's a human raised on the logical planet Vulcan by Spock's father.  
	Burnham is intelligent and struggles to balance her human emotions with Vulcan logic.  
	She's become a Starfleet captain known for her determination and problem-solving skills.
	Originally played by actress Sonequa Martin-Green`,

	`James T. Kirk, also known as Captain Kirk, is a fictional character from the Star Trek franchise.  
	He's the iconic captain of the starship USS Enterprise, 
	boldly exploring the galaxy with his crew.  
	Originally played by actor William Shatner, 
	Kirk has appeared in TV series, movies, and other media.`,

	`Jean-Luc Picard is a fictional character in the Star Trek franchise.
	He's most famous for being the captain of the USS Enterprise-D,
	a starship exploring the galaxy in the 24th century.
	Picard is known for his diplomacy, intelligence, and strong moral compass.
	He's been portrayed by actor Patrick Stewart.`,
}

func TestEmbeddingCreation(t *testing.T) {
	ollamaUrl := "http://localhost:11434"
	model := "all-minilm"

	content := `The best pizza of the world is the pineapple pizza`

	query := Query4Embedding{
		Model:  model,
		Prompt: content,
	}

	vector, err := CreateEmbedding(ollamaUrl, query, "000")

	if err != nil {
		t.Fatal("😡:", err)
	}

	if vector.Embedding[0] == -0.28555548191070557 && vector.Embedding[1] == 0.11453928798437119 && vector.Embedding[2] == 0.04224501550197601 {
		log.Println("🙂", vector)
	} else {
		t.Fatal("😡:", err)
	}

}

func TestVectorStore(t *testing.T) {

	ollamaUrl := "http://localhost:11434"
	embeddingsModel := "all-minilm:22m"

	store := MemoryVectorStore{
		Records: make(map[string]VectorRecord),
	}

	// Create embeddings from documents and save them in the store
	for idx, doc := range docs {
		fmt.Println("Creating embedding from document ", idx)
		embedding, err := CreateEmbedding(
			ollamaUrl,
			Query4Embedding{
				Model:  embeddingsModel,
				Prompt: doc,
			},
			strconv.Itoa(idx),
		)
		if err != nil {
			t.Fatal("😡:", err)
		} else {
			_, err := store.Save(embedding)

			if err != nil {
				t.Fatal("😡:", err)
			}
		}
	}

	vectors, err := store.GetAll()
	if err != nil {
		t.Fatal("😡:", err)
	}
	if len(vectors) == 3 {
		log.Println("🙂", vectors)
	} else {
		t.Fatal("😡:", err)
	}

}

func TestSimilaritySearch(t *testing.T) {

	ollamaUrl := "http://localhost:11434"
	embeddingsModel := "all-minilm:22m"

	store := MemoryVectorStore{
		Records: make(map[string]VectorRecord),
	}

	// Create embeddings from documents and save them in the store
	for idx, doc := range docs {
		fmt.Println("Creating embedding from document ", idx)
		embedding, err := CreateEmbedding(
			ollamaUrl,
			Query4Embedding{
				Model:  embeddingsModel,
				Prompt: doc,
			},
			strconv.Itoa(idx),
		)
		if err != nil {
			t.Fatal("😡:", err)
		} else {
			_, err := store.Save(embedding)

			if err != nil {
				t.Fatal("😡:", err)
			}
		}
	}

	userContent := `Who is Jean-Luc Picard?`
	//userContent := `Who is Philippe Charrière?`

	// Create an embedding from the question
	embeddingFromQuestion, err := CreateEmbedding(
		ollamaUrl,
		Query4Embedding{
			Model:  embeddingsModel,
			Prompt: userContent,
		},
		"question",
	)
	if err != nil {
		log.Fatalln("😡:", err)
	}

	similarities, err := store.SearchSimilarities(embeddingFromQuestion, 0.4)

	if err != nil {
		t.Fatal("😡:", err)
	}
	if len(similarities) == 0 {
		t.Fatal("😡 no similarity:", err)
	} else {
		log.Println("🙂", similarities)
	}

}

func TestJAccardSimilarity(t *testing.T) {
	userContent := `Who is Jean-Luc Picard?`

	splittedUserContent := strings.Fields(userContent)

	type similarity struct {
		coeff float64
		doc   string
	}

	similarities := []similarity{}

	// Calculate Jaccard index for every document
	// the highest index is related to the best similarity
	for idx, doc := range docs {
		jaccardIndex := JaccardSimilarityCoeff(splittedUserContent, strings.Fields(doc))
		fmt.Println("- 📝", idx, "Jaccard index:", jaccardIndex)

		similarity := similarity{
			coeff: jaccardIndex,
			doc:   docs[idx],
		}

		if jaccardIndex > 0.03 {
			similarities = append(similarities, similarity)
		}

	}
	if len(similarities) == 0 {
		t.Fatal("😡 no similarity:", errors.New("no similarity"))
	} else {
		log.Println("🙂 Similarities:", similarities)
	}
}
