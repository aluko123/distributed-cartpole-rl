package cartpole

import (
	"math"
	"math/rand"
)

const (
	gravity        = 9.81
	massCart       = 1.0
	massPole       = 0.1
	length         = 0.5
	totalMass      = massCart + massPole
	poleMassLength = massPole * length
	forceMax       = 10.0
	tau            = 0.02

	xThreshold     = 2.4
	thetaThreshold = 12.0 * math.Pi / 180.0
	maxSteps       = 500
)

type State struct {
	X        float64 `json:"x"`
	XDot     float64 `json:"x_dot"`
	Theta    float64 `json:"theta"`
	ThetaDot float64 `json:"theta_dot"`
}

type Env struct {
	State State
	Steps int
	Rand  *rand.Rand
}

func NewEnv(rng *rand.Rand) *Env {
	if rng == nil {
		rng = rand.New(rand.NewSource(rand.Int63()))
	}
	env := &Env{Rand: rng}
	env.Reset()
	return env
}

func (e *Env) Reset() State {
	e.State = State{
		X:        e.Rand.Float64()*0.1 - 0.05,
		XDot:     e.Rand.Float64()*0.1 - 0.05,
		Theta:    e.Rand.Float64()*0.1 - 0.05,
		ThetaDot: e.Rand.Float64()*0.1 - 0.05,
	}
	e.Steps = 0
	return e.State
}

func (e *Env) Step(action int) (State, float64, bool) {
	force := forceMax
	if action == 0 {
		force = -forceMax
	}

	x := e.State.X
	xDot := e.State.XDot
	theta := e.State.Theta
	thetaDot := e.State.ThetaDot

	cosTheta := math.Cos(theta)
	sinTheta := math.Sin(theta)

	temp := (force + poleMassLength*thetaDot*thetaDot*sinTheta) / totalMass
	thetaAcc := (gravity*sinTheta - cosTheta*temp) / (length * (4.0/3.0 - massPole*cosTheta*cosTheta/totalMass))
	xAcc := temp - poleMassLength*thetaAcc*cosTheta/totalMass
	x += tau * xDot
	xDot += tau * xAcc
	theta += tau * thetaDot
	thetaDot += tau * thetaAcc

	e.State = State{
		X:        x,
		XDot:     xDot,
		Theta:    theta,
		ThetaDot: thetaDot,
	}
	e.Steps++

	done := x < -xThreshold || x > xThreshold || theta < -thetaThreshold || theta > thetaThreshold || e.Steps >= maxSteps
	reward := 1.0
	if done && e.Steps < maxSteps {
		reward = 0.0
	}
	return e.State, reward, done
}

func MaxSteps() int {
	return maxSteps
}
