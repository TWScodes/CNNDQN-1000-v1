// ** This DQN class is designed to work with the neural network in the 'main.js' file, so it may have some diferences! **
class DQN {
    constructor(stateShape, actionSpaceSize, learningRate=0.001, discountFactor=0.99, explorationRate=1.0, explorationDecayRate=0.001, minExplorationRate=0.01, replayBufferSize=10000, batchSize=32) {
        this.stateShape = stateShape; // Shape of the game state input (e.g., image dimensions and channels)
        this.actionSpaceSize = actionSpaceSize; // Number of possible actions
        this.learningRate = learningRate;
        this.discountFactor = discountFactor;
        this.explorationRate = explorationRate;
        this.explorationDecayRate = explorationDecayRate;
        this.minExplorationRate = minExplorationRate;
        this.replayBuffer = new ReplayBuffer(replayBufferSize);
        this.batchSize = batchSize;

        // --- Q-Network (using CNN) ---
        this.qNetwork = new CNN([
            new Conv2DLayer(32, [8, 8], 4, 'relu', stateShape), // Example layers - adjust as needed
            new Conv2DLayer(64, [4, 4], 2, 'relu'),
            new Conv2DLayer(64, [3, 3], 1, 'relu'),
            new FlattenLayer(),
            new DenseLayer(512, 'relu'),
            new DenseLayer(actionSpaceSize, 'linear') // Output layer - linear activation for Q-values
        ]);
        this.targetNetwork = cloneNetwork(this.qNetwork); // Target network (initially same as Q-network)
        this.updateTargetNetworkFrequency = 1000; // Example frequency to update target network weights
        this.stepCount = 0; // To track steps for target network updates and exploration decay
    }

    getAction(state) { // State: [state_height, state_width, state_channels] - needs to be in correct format for CNN input
        if (Math.random() < this.explorationRate) {
            return Math.floor(Math.random() * this.actionSpaceSize); // Explore - choose random action
        } else {
            const qValues = this.qNetwork.forward(state); // Get Q-values from Q-network for current state
            return argMax(qValues); // Exploit - choose action with highest Q-value
        }
    }

    storeTransition(state, action, reward, nextState, done) {
        this.replayBuffer.add(state, action, reward, nextState, done);
    }

    train() {
        if (this.replayBuffer.size() < this.batchSize) {
            return; // Not enough samples in replay buffer to train
        }

        const batch = this.replayBuffer.sample(this.batchSize);
        const states = batch.states;
        const actions = batch.actions;
        const rewards = batch.rewards;
        const nextStates = batch.nextStates;
        const dones = batch.dones;

        // --- 1. Predict Q-values for current states (using Q-network) ---
        const qValuesCurrentState = this.qNetwork.forward(states); // Shape: [batch_size, actionSpaceSize]

        // --- 2. Predict Q-values for next states (using Target network) ---
        const qValuesNextStateTarget = this.targetNetwork.forward(nextStates); // Shape: [batch_size, actionSpaceSize]

        // --- 3. Calculate Target Q-values (using Bellman equation) ---
        let qTargets = clone(qValuesCurrentState); // Initialize with current Q-values, we will update based on actions taken
        for (let i = 0; i < this.batchSize; i++) {
            const bestNextActionQValue = Math.max(...qValuesNextStateTarget[i]); // Max Q-value for next state
            qTargets[i][actions[i]] = rewards[i] + (dones[i] ? 0 : this.discountFactor * bestNextActionQValue); // Bellman equation
        }

        // --- 4. Calculate Loss (Mean Squared Error) ---
        const loss = meanSquaredError(qValuesCurrentState, qTargets); // Loss between predicted Q-values and target Q-values

        // --- 5. Backpropagation and Optimization (Train Q-Network) ---
        this.qNetwork.backward(meanSquaredErrorDerivative(qValuesCurrentState, qTargets)); // Backpropagate loss
        this.qNetwork.applyGradients(this.learningRate); // Apply gradients to Q-network weights

        // --- 6. Update Target Network (Periodically) ---
        this.stepCount++;
        if (this.stepCount % this.updateTargetNetworkFrequency === 0) {
            this.targetNetwork = cloneNetwork(this.qNetwork); // Sync target network weights from Q-network
            console.log("Target network updated");
        }

        // --- 7. Exploration Rate Decay ---
        this.explorationRate = Math.max(this.minExplorationRate, this.explorationRate - this.explorationDecayRate);
        return loss; // Return loss for monitoring (optional)
    }


    loadWeights(weights) { // weights should be an array of tensors in the order of layers
        let paramIndex = 0;
        for (let layer of this.qNetwork.layers) {
            let layerParams = layer.getParameters();
            for (let i = 0; i < layerParams.length; i++) {
                layerParams[i] = weights[paramIndex++]; // Directly assign loaded weights
            }
        }
        this.targetNetwork = cloneNetwork(this.qNetwork); // Sync target network weights after loading
    }

    getWeights() { // Get all weights of Q-network (for saving/loading)
        return this.qNetwork.getParameters();
    }


    summary() {
        console.log("--- DQN Summary ---");
        console.log("Q-Network Architecture:");
        this.qNetwork.summary();
        console.log("\nDQN Parameters:");
        console.log(`Learning Rate: ${this.learningRate}, Discount Factor: ${this.discountFactor}`);
        console.log(`Exploration Rate: ${this.explorationRate.toFixed(3)}, Decay Rate: ${this.explorationDecayRate}, Min Rate: ${this.minExplorationRate}`);
        console.log(`Replay Buffer Size: ${this.replayBuffer.capacity}, Batch Size: ${this.batchSize}`);
        console.log(`Target Network Update Frequency: ${this.updateTargetNetworkFrequency}`);
        console.log("--- End DQN Summary ---");
    }
}
