package agent;

import org.apache.commons.math3.linear.RealVector;

public class Action {
    private RealVector action;

    public Action(RealVector action) {
        this.action = action;
    }

    public RealVector getAction() {
        return action;
    }

    public void setAction(RealVector action) {
        this.action = action;
    }
}
