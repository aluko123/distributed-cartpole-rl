package worker

import (
	"math"
	"math/rand"
)

type PolicyWeights struct {
	W  [][]float64 `json:"w"`  // shape: [2][4]
	B  []float64   `json:"b"`  // shape: [2]
	VW []float64   `json:"vw"` // shape: [4]
	VB float64     `json:"vb"`
}

type Policy struct {
	Weights PolicyWeights
}

func DefaultWeights() PolicyWeights {
	return PolicyWeights{
		W: [][]float64{
			{0.01, 0.01, 0.01, 0.01},
			{-0.01, -0.01, -0.01, -0.01},
		},
		B:  []float64{0, 0},
		VW: []float64{0, 0, 0, 0},
		VB: 0,
	}
}

func NewPolicy(weights PolicyWeights) *Policy {
	return &Policy{
		Weights: weights,
	}
}

// Action returns chosen action, log-probability, and value estimate
func (p *Policy) Action(state []float64, rng *rand.Rand) (int, float64, float64) {
	logits := make([]float64, 2)
	for i := 0; i < 2; i++ {
		logits[i] = p.Weights.B[i]
		for j := 0; j < len(state); j++ {
			logits[i] += p.Weights.W[i][j] * state[j]
		}
	}
	probs := softmax(logits)
	choice := sampleCategorical(probs, rng)
	logProb := math.Log(probs[choice] + 1e-8)
	value := p.Weights.VB

	for j := 0; j < len(state); j++ {
		value += p.Weights.VW[j] * state[j]
	}

	return choice, logProb, value
}

func softmax(logits []float64) []float64 {
	maxLogit := logits[0]
	for _, v := range logits[1:] {
		if v > maxLogit {
			maxLogit = v
		}
	}
	values := make([]float64, len(logits))
	var sum float64
	for i, v := range logits {
		values[i] = math.Exp(v - maxLogit)
		sum += values[i]
	}
	for i := range values {
		values[i] /= sum
	}
	return values
}

func sampleCategorical(probs []float64, rng *rand.Rand) int {
	threshold := rng.Float64()
	var cumulativeProb float64
	for i, prob := range probs {
		cumulativeProb += prob
		if threshold <= cumulativeProb {
			return i
		}
	}
	return len(probs) - 1
}
