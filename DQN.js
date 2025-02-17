// ** This is an unrelated DQN class that should work independantly. **
wait (()=>{document.head.appendChild(Object.assign(document.createElement("script"),{src:"https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.0.0/dist/tf.min.js"}))})();

class DQN {
  constructor(stateSize, actionSize, config = {}) {
    this.stateSize = stateSize;
    this.actionSize = actionSize;
    this.learningRate = config.learningRate || 0.001; // Initial Learning Rate (for hyperparameter tuning)
    this.gamma = config.gamma || 0.99;
    // Epsilon related parameters are REMOVED for pure Boltzmann exploration
    // this.epsilon = config.epsilon || 1.0;
    // this.epsilonMin = config.epsilonMin || 0.01;
    // this.epsilonDecay = config.epsilonDecay || 0.995;
    // this.epsilonDecayRate = config.epsilonDecayRate || 0.001;
    this.batchSize = config.batchSize || 64;
    this.updateTargetFreq = config.updateTargetFreq || 1000;
    this.learnStart = config.learnStart || 1000;
    this.maxMemory = config.maxMemory || 10000;
    this.temperature = config.temperature || 1.0; // Boltzmann exploration temperature - now primary exploration control

    this.initialLearningRate = config.initialLearningRate || 0.001; // For learning rate decay
    this.decayRate = config.decayRate || 0.96;
    this.decaySteps = config.decaySteps || 1000;
    this.temperatureDecayRate = config.temperatureDecayRate || 0.999; // Temperature decay

    this.units_dense_layer_1 = config.units_dense_layer_1 || 128; // Increased units in layers
    this.units_dense_layer_2 = config.units_dense_layer_2 || 128;
    this.units_dense_layer_3 = config.units_dense_layer_3 || 128;


    // Q-network and target network (for Double DQN)
    this.qNetwork = this.buildModel();
    this.targetNetwork = this.buildModel();
    this.targetNetwork.setWeights(this.qNetwork.getWeights());

    // Optimizer with Learning Rate Scheduling (Exponential Decay)
    this.optimizer = tf.train.adam(this.initialLearningRate);
    this.learningRateSchedule = (step) => this.initialLearningRate * Math.pow(this.decayRate, Math.floor(step / this.decaySteps));


    // Experience Replay
    this.memory = [];
    this.steps = 0;
  }

  buildModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({
      units: this.units_dense_layer_1, // Increased units and hyperparameter
      inputShape: [this.stateSize],
      activation: 'relu'
    }));
    model.add(tf.layers.batchNormalization()); // Batch Normalization
    model.add(tf.layers.dense({
      units: this.units_dense_layer_2, // Increased units and hyperparameter
      activation: 'relu'
    }));
    model.add(tf.layers.batchNormalization()); // Batch Normalization
    model.add(tf.layers.dense({ // Added one more dense layer
      units: this.units_dense_layer_3, // Increased units and hyperparameter
      activation: 'relu'
    }));
    model.add(tf.layers.batchNormalization()); // Batch Normalization
    model.add(tf.layers.dense({
      units: this.actionSize,
      activation: 'linear' // Output Q-values for each action
    }));
    model.compile({ optimizer: this.optimizer, loss: tf.losses.huberLoss }); // Huber Loss
    return model;
  }

  act(state) {
    const stateTensor = tf.tensor2d([state]);
    const qValues = this.qNetwork.predict(stateTensor);

    // Pure Boltzmann Exploration (Softmax) - Temperature driven exploration
    const probabilities = tf.softmax(qValues.div(this.temperature)).dataSync();
    let cumulativeProbability = 0;
    const randomNumber = Math.random();
    for (let i = 0; i < this.actionSize; i++) {
        cumulativeProbability += probabilities[i];
        if (randomNumber <= cumulativeProbability) {
            stateTensor.dispose();
            return i; // Select action based on probabilities
        }
    }
    stateTensor.dispose();
    return -1; // Should ideally not reach here, but return -1 as a fallback. Consider throwing error for debugging.
  }

  remember(state, action, reward, nextState, done) {
    this.memory.push({ state, action, reward, nextState, done });
    if (this.memory.length > this.maxMemory) {
      this.memory.shift(); // Remove the oldest experience
    }
  }

  async train() {
    if (this.memory.length < this.learnStart) return;

    const batch = this.sampleBatch();
    const states = batch.map(item => item.state);
    const actions = batch.map(item => item.action);
    const rewards = batch.map(item => item.reward);
    const nextStates = batch.map(item => item.nextState);
    const dones = batch.map(item => item.done);

    const stateTensor = tf.tensor2d(states);
    const nextStateTensor = tf.tensor2d(nextStates);

    const qValues = this.qNetwork.predict(stateTensor);
    const nextQValues = this.targetNetwork.predict(nextStateTensor);
    const maxNextQ = nextQValues.max(-1).dataSync();

    // Update Q-values using Double DQN
    const targetQValues = qValues.clone();
    for (let i = 0; i < batch.length; i++) {
      const target = rewards[i] + (this.gamma * maxNextQ[i] * (1 - dones[i]));
      targetQValues.arraySync()[i][actions[i]] = target;
    }

        // Learning Rate Scheduling - Apply decayed learning rate before each training step
        const currentLearningRate = this.learningRateSchedule(this.steps);
        this.optimizer.learningRate = currentLearningRate;


    // Perform the gradient update step
    await this.qNetwork.fit(stateTensor, targetQValues, { epochs: 1, batchSize: this.batchSize });
    
    stateTensor.dispose();
    nextStateTensor.dispose();

    // Temperature decay - for pure Boltzmann exploration, decay temperature instead of epsilon
    if (this.temperature > 0.01) { // Example minimum temperature - tune this
        this.temperature *= this.temperatureDecayRate;
    }


    // Periodically update the target network
    if (this.steps % this.updateTargetFreq === 0) {
      this.targetNetwork.setWeights(this.qNetwork.getWeights());
    }
    this.steps++;
  }

  sampleBatch() {
    const batchSize = Math.min(this.memory.length, this.batchSize);
    const batch = [];
    for (let i = 0; i < batchSize; i++) {
      const index = Math.floor(Math.random() * this.memory.length);
      batch.push(this.memory[index]);
    }
    return batch;
  }

  getEpsilon() { // getEpsilon is now misleading as epsilon is removed, rename to getTemperature for clarity if needed
    return this.temperature; // return temperature instead of epsilon
    }
}
