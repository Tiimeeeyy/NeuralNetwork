package agent;

import math.q_calculations.QCalculations;
import network.NeuralNetwork;
import network.ReplayBuffer;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

import java.util.List;
import java.util.Random;

public class QLearningAgent {
    // TODO: Finish implementations.
    // TODO: Add tests.
    // TODO: Connect to game engine.
    // TODO: Handle episodes.
    // TODO: Training loop.
    // TODO: Hyperparameter (Report).
    // TODO: Save the model to disk (Optional).
    // TODO: Save the trained model / Save the training.
    // TODO: Visualisation.
    private static final int NUM_ANGLES = 100; // The amount of angles we let the agent explore.
    private static final int NUM_POWERS = 5; // The amount of shot powers the agent can use (In our case 0-5 m/s).
    private static final double EPSILON_DECAY = 0.995;
    private static final double MIN_EPSILON = 0.01;
    private NeuralNetwork qNetwork;
    private ReplayBuffer replayBuffer;
    private double epsilon;
    private Random random;
    private QCalculations qCalculations;

    public QLearningAgent(NeuralNetwork qNetwork, ReplayBuffer replayBuffer, double epsilon, QCalculations qCalculations) {

        this.qNetwork = qNetwork;
        this.replayBuffer = replayBuffer;
        this.epsilon = epsilon;
        this.qCalculations = qCalculations;
    }

    public RealVector chooseAction(State state) {
        int i = 1;
        if (random.nextDouble() < epsilon) {
            if (random.nextDouble() < 0.5) {
                i = -1;
            }
            double randomPower = 1 + random.nextDouble() * 4;
            return new ArrayRealVector(new double[]{random.nextDouble() * i, random.nextDouble() * i, random.nextDouble(), randomPower});
        } else {
            return getBestAction(state);
        }
    }

    private RealVector getBestAction(State state) {
        double maxQValue = Double.NEGATIVE_INFINITY;
        RealVector bestAction = null;

        for (int angle = 0; angle < NUM_ANGLES; angle++) {
            for (double power = 0; power <= NUM_POWERS; power += 0.2) {
                double rad = Math.toRadians(angle);
                double xDir = Math.cos(rad);
                double yDir = Math.sin(rad);

                RealVector action = new ArrayRealVector(new double[]{xDir, yDir, power});
                // TODO: Connect the Null value to the game engine!!
                double qValue = qCalculations.calculateQValue(state, new Action(action), 0, null);

                if (qValue > maxQValue) {
                    maxQValue = qValue;
                    bestAction = action;
                }
            }

        }
        return bestAction;
    }

    public void updateQValues() {

        List<ReplayBuffer.Experience> batch = ReplayBuffer.sampleBatch(4);
        for (ReplayBuffer.Experience experience : batch) {
            double targetQ = qCalculations.calculateQValue(
                    experience.getState(),
                    experience.getAction(),
                    experience.getReward(),
                    experience.getNextState()
            );
            qNetwork.train(
                    experience.getState().getCurrentPosition().append(experience.getAction().getAction()), new ArrayRealVector(new double[]{targetQ}), 0.01
            );
        }
    }

    public void addExperienceToBuffer(State state, Action action, double reward, State nextState) {
        replayBuffer.addExperience(new ReplayBuffer.Experience(state, action, reward, nextState));
    }

    public void decayEpsilon() {
        epsilon = Math.max(MIN_EPSILON, epsilon * EPSILON_DECAY);
    }
}
