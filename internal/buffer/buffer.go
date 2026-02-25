package buffer

import (
	"errors"
	"sync"
	"time"
)

type Item struct {
	Trajectory Trajectory
	EnqueuedAt time.Time
}

type ReplayBuffer struct {
	mu       sync.Mutex
	items    []Item
	capacity int
	policy   string // "fifo" or "freshness"
}

var (
	ErrBufferFull  = errors.New("buffer is full")
	ErrBufferEmpty = errors.New("buffer is empty")
)

func NewReplayBuffer(capacity int, policy string) (*ReplayBuffer, error) {
	if capacity <= 0 {
		return nil, errors.New("capacity must be greater than zero")
	}
	if policy != "fifo" && policy != "freshness" {
		return nil, errors.New("policy must be 'fifo' or 'freshness'")
	}
	return &ReplayBuffer{
		items:    make([]Item, 0, capacity),
		capacity: capacity,
		policy:   policy,
	}, nil
}

func (rb *ReplayBuffer) Enqueue(item Item) error {
	rb.mu.Lock()
	defer rb.mu.Unlock()

	if len(rb.items) >= rb.capacity {
		return ErrBufferFull
	}
	rb.items = append(rb.items, item)
	return nil
}

func (rb *ReplayBuffer) Dequeue() (Item, error) {
	rb.mu.Lock()
	defer rb.mu.Unlock()

	if len(rb.items) == 0 {
		return Item{}, ErrBufferEmpty
	}

	switch rb.policy {
	case "fifo":
		item := rb.items[0]
		rb.items = rb.items[1:]
		return item, nil
	case "freshness":
		item := rb.items[len(rb.items)-1]
		rb.items = rb.items[:len(rb.items)-1]
		return item, nil
	default:
		return Item{}, errors.New("unknown policy")
	}
}

func (rb *ReplayBuffer) Capacity() int {
	return rb.capacity
}

func (rb *ReplayBuffer) Policy() string {
	rb.mu.Lock()
	defer rb.mu.Unlock()

	return rb.policy
}

func (rb *ReplayBuffer) SetPolicy(policy string) error {
	if policy != "fifo" && policy != "freshness" {
		return errors.New("policy must be 'fifo' or 'freshness'")
	}

	rb.mu.Lock()
	defer rb.mu.Unlock()

	rb.policy = policy
	return nil
}

func (rb *ReplayBuffer) Size() int {
	rb.mu.Lock()
	defer rb.mu.Unlock()

	return len(rb.items)
}
