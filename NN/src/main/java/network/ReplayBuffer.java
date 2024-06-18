package network;

import org.apache.commons.math3.linear.RealVector;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * This class is used to save the Experiences of the Artificial neural network.
 */
public class ReplayBuffer {
    private static final int MAX_SIZE = 10000;
    private static List<Experience> experiences = new ArrayList<>();


    // Nested static class to make sure the objects are immutable.
    public static class Experience {
        public RealVector state;
        public RealVector action;
        public double reward;
        public RealVector nextState;

        public Experience(RealVector state, RealVector action, double reward, RealVector nextState) {
            this.state = state;
            this.action = action;
            this.reward = reward;
            this.nextState = nextState;
        }

        /**
         * Creates a random batch of experiences out of the List in memory.
         * @param batchSize The size of the batch.
         * @return A list containing random experiences from the list in memory.
         */
        public static List<Experience> sampleBatch(int batchSize) {

            batchSize = Math.min(batchSize, experiences.size());

            List<Experience> batch = new ArrayList<>();

            Collections.shuffle(experiences);

            for (int i = 0; i < batchSize; i++) {
                batch.add(experiences.get(i));
            }
            return batch;
        }

        /**
         * Adds an experience to the list ("to memory")
         * @param experience The experience to be added.
         */
        public void addExperience(Experience experience) {
            if (experiences.size() >= MAX_SIZE) {
                experiences.remove(0);
            }
            experiences.add(experience);
        }


    }
}

