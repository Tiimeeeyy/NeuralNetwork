package agent;

import org.apache.commons.math3.linear.RealVector;

public class State {
    private RealVector currentPosition;


    public State(RealVector currentPosition) {
        this.currentPosition = currentPosition;
    }

    public RealVector getCurrentPosition() {
        return currentPosition;
    }

    public void setCurrentPosition(RealVector currentPosition) {
        this.currentPosition = currentPosition;
    }
}
