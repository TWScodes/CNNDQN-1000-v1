// --- Matrix/Tensor Operations ---

function zeros(shape) {
    let result = [];
    if (shape.length === 1) {
        return Array(shape[0]).fill(0);
    } else {
        let currentShape = shape.slice(1);
        for (let i = 0; i < shape[0]; i++) {
            result.push(zeros(currentShape));
        }
        return result;
    }
}

function random(shape, scale = 1.0) { // Scaled random for initialization
    let result = [];
    if (shape.length === 1) {
        return Array.from({ length: shape[0] }, () => (Math.random() * 2 - 1) * scale); // -1 to 1 range
    } else {
        let currentShape = shape.slice(1);
        for (let i = 0; i < shape[0]; i++) {
            result.push(random(currentShape, scale));
        }
        return result;
    }
}

function add(tensor1, tensor2) {
    if (!areShapesEqual(shape(tensor1), shape(tensor2))) {
        throw new Error("Tensor shapes must be equal for addition.");
    }
    return elementWiseOperation(tensor1, tensor2, (a, b) => a + b);
}

function subtract(tensor1, tensor2) {
    if (!areShapesEqual(shape(tensor1), shape(tensor2))) {
        throw new Error("Tensor shapes must be equal for subtraction.");
    }
    return elementWiseOperation(tensor1, tensor2, (a, b) => a - b);
}

function multiply(tensor1, tensor2) { // Element-wise multiplication
    if (!areShapesEqual(shape(tensor1), shape(tensor2))) {
        throw new Error("Tensor shapes must be equal for element-wise multiplication.");
    }
    return elementWiseOperation(tensor1, tensor2, (a, b) => a * b);
}

function scalarMultiply(tensor, scalar) {
    return elementWiseOperation(tensor, scalar, (a, s) => a * s);
}

function scalarAdd(tensor, scalar) {
    return elementWiseOperation(tensor, scalar, (a, s) => a + s);
}

function matrixMultiply(matrix1, matrix2) { // Optimized 2D matrix multiplication
    const shape1 = shape(matrix1);
    const shape2 = shape(matrix2);

    if (shape1[1] !== shape2[0]) {
        throw new Error(`Matrices dimensions mismatch for matrix multiplication: (${shape1[0]}x${shape1[1]}) and (${shape2[0]}x${shape2[1]})`);
    }

    const rows1 = shape1[0];
    const cols1 = shape1[1];
    const cols2 = shape2[1];

    let result = zeros([rows1, cols2]);

    for (let i = 0; i < rows1; i++) {
        for (let k = 0; k < cols1; k++) { // Optimized loop order for potential cache locality
            const val1 = matrix1[i][k]; // Store matrix1[i][k] for reuse in inner loop
            for (let j = 0; j < cols2; j++) {
                result[i][j] += val1 * matrix2[k][j];
            }
        }
    }
    return result;
}

function transpose(matrix) { // For 2D matrices
    const rows = matrix.length;
    const cols = matrix[0].length;
    let result = zeros([cols, rows]);
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}

function sum(tensor) { // Sum of all elements in a tensor
    let s = 0;
    traverseTensor(tensor, (val) => { s += val; });
    return s;
}

function mean(tensor) {
    return sum(tensor) / size(tensor);
}

function reshape(tensor, newShape) { // Basic reshape (assuming compatible size)
    let flatTensor = [];
    traverseTensor(tensor, (val) => flatTensor.push(val));
    if (flatTensor.length !== size(newShape)) {
        throw new Error("Reshape size is incompatible.");
    }
    return arrayToTensor(flatTensor, newShape);
}

function arrayToTensor(arr, shapeArray) { // Helper for reshape
    if (shapeArray.length === 0) {
        return arr.shift(); // Base case: single value
    }
    const subShape = shapeArray.slice(1);
    const count = shapeArray[0];
    let result = [];
    for (let i = 0; i < count; i++) {
        result.push(arrayToTensor(arr, subShape));
    }
    return result;
}


// --- Utility Functions ---

function shape(tensor) {
    if (!Array.isArray(tensor)) {
        return []; // Scalar has no shape
    }
    let currentShape = [tensor.length];
    let firstElement = tensor[0];
    while (Array.isArray(firstElement)) {
        currentShape.push(firstElement.length);
        firstElement = firstElement[0]; // Assuming consistent shapes
    }
    return currentShape;
}

function size(shapeArray) {
    if (shapeArray.length === 0) return 1; //Scalar size is 1
    return shapeArray.reduce((prod, dim) => prod * dim, 1);
}

function elementWiseOperation(tensor1, tensor2, operation) {
    let result = [];
    if (!Array.isArray(tensor1)) { //Scalar case
        return operation(tensor1, tensor2); // Assuming tensor2 is also scalar or operation handles scalar
    }

    for (let i = 0; i < tensor1.length; i++) {
        if (Array.isArray(tensor1[i])) {
            result.push(elementWiseOperation(tensor1[i], tensor2[i], operation)); // Recursive for nested arrays
        } else {
            result.push(operation(tensor1[i], tensor2[i])); // Apply operation to elements
        }
    }
    return result;
}

function traverseTensor(tensor, callback) { // For iterating over all elements
    if (!Array.isArray(tensor)) {
        callback(tensor); // Base case: apply callback to element
        return;
    }
    for (let i = 0; i < tensor.length; i++) {
        traverseTensor(tensor[i], callback); // Recursive call for sub-arrays
    }
}

function areShapesEqual(shape1, shape2) {
    if (shape1.length !== shape2.length) return false;
    for (let i = 0; i < shape1.length; i++) {
        if (shape1[i] !== shape2[i]) return false;
    }
    return true;
}

function clone(tensor) { // Deep clone a tensor
    return JSON.parse(JSON.stringify(tensor));
}

function printTensor(tensor, indent = 0) {
    if (!Array.isArray(tensor)) {
        process.stdout.write(tensor.toFixed(3) + ", "); // Print numbers formatted
        return;
    }
    if (tensor.length === 0) {
        process.stdout.write("[], ");
        return;
    }

    process.stdout.write("[\n");
    for (let i = 0; i < tensor.length; i++) {
        process.stdout.write("  ".repeat(indent + 1)); // Indentation
        printTensor(tensor[i], indent + 1);
        if (i === tensor.length - 1) { // No comma after last element in this dimension
            process.stdout.write("\n");
        }
    }
    process.stdout.write("  ".repeat(indent) + "], ");
}

// --- Activation Functions and their Derivatives ---

const activations = {
    relu: (x) =>  !Array.isArray(x) ? Math.max(0, x) : elementWiseOperation(x, 0, (val, _) => Math.max(0, val)),
    reluDerivative: (x) => !Array.isArray(x) ? (x > 0 ? 1 : 0) : elementWiseOperation(x, 0, (val, _) => val > 0 ? 1 : 0),
    sigmoid: (x) => !Array.isArray(x) ? 1 / (1 + Math.exp(-x)) : elementWiseOperation(x, 0, (val, _) => 1 / (1 + Math.exp(-val))),
    sigmoidDerivative: (output) => !Array.isArray(output) ? output * (1 - output) : elementWiseOperation(output, 0, (val, _) => val * (1 - val)),
    tanh: (x) => !Array.isArray(x) ? Math.tanh(x) : elementWiseOperation(x, 0, (val, _) => Math.tanh(val)),
    tanhDerivative: (output) => !Array.isArray(output) ? 1 - output * output : elementWiseOperation(output, 0, (val, _) => 1 - val * val),
    linear: (x) => x,
    linearDerivative: (x) => !Array.isArray(x) ? 1 : elementWiseOperation(x, 0, () => 1)
};

// --- Layers ---

class Layer { // Base class for layers (might not be strictly necessary, but good for structure)
    constructor() {
        this.outputShape = undefined; // To be defined by subclasses
        this.inputShape = undefined;
    }

    forward(input) {
        throw new Error("Forward method must be implemented in subclass.");
    }

    backward(outputGradient) {
        throw new Error("Backward method must be implemented in subclass.");
    }

    getParameters() { // For layers with weights/biases
        return []; // Default: no parameters
    }

    getGradients() {
        return []; // Default: no gradients
    }

    applyGradients(learningRate) { // Default: no parameters to update
    }
}


class DenseLayer extends Layer {
    constructor(units, activation='relu', inputShape=null) { // units: number of neurons in this layer
        super();
        this.units = units;
        this.activationName = activation;
        this.activation = activations[activation];
        this.activationDerivative = activations[activation + 'Derivative'];
        this.weights = null; // Initialized in `build`
        this.biases = null;
        this.output = null; // Store output from forward pass for backprop
        this.input = null;
        this.weightsGradient = null;
        this.biasesGradient = null;

        if (inputShape) {
            this.build(inputShape);
        }
    }

    build(inputShape) { // 'inputShape' is the shape of the input tensor [input_size] for Dense
        this.inputShape = inputShape;
        this.outputShape = [this.units]; // Output shape of Dense is always [units]
        this.weights = random([this.units, this.inputShape[0]], Math.sqrt(2.0/this.inputShape[0])); // Kaiming/He initialization
        this.biases = zeros([this.units, 1]); // Biases as column vector
    }


    forward(input) { // Input is expected to be a column vector [input_size, 1]
        if (!this.weights) { // Lazy initialization if inputShape was not provided in constructor
            this.build(shape(input));
        }
        this.input = input; // Store input for backprop
        const z = add(matrixMultiply(this.weights, input), this.biases); // z = W*a + b
        this.output = this.activation(z); // a' = activation(z)
        return this.output;
    }

    backward(outputGradient) { // outputGradient from the layer above
        // outputGradient shape: [units, 1] (same as this layer's output)

        // 1. Derivative of activation function at the current layer's output
        const activationDerivativeVal = this.activationDerivative(this.output); // Shape: [units, 1]

        // 2. Element-wise multiplication of outputGradient and activation derivative
        const delta = multiply(outputGradient, activationDerivativeVal); // Delta (error signal for this layer) Shape: [units, 1]

        // 3. Gradients for weights and biases
        this.weightsGradient = matrixMultiply(delta, transpose(this.input)); // dW = delta * a_prev.T  Shape: [units, input_size]
        this.biasesGradient = delta; // db = delta  Shape: [units, 1]

        // 4. Gradient to pass to the previous layer (input gradient)
        const inputGradient = matrixMultiply(transpose(this.weights), delta); // Gradient w.r.t. input  Shape: [input_size, 1]

        return inputGradient; // Pass gradient to the previous layer
    }

    getParameters() {
        return [this.weights, this.biases];
    }

    getGradients() {
        return [this.weightsGradient, this.biasesGradient];
    }

    applyGradients(learningRate) {
        this.weights = subtract(this.weights, scalarMultiply(this.weightsGradient, learningRate));
        this.biases = subtract(this.biases, scalarMultiply(this.biasesGradient, learningRate));
    }
}

// --- Convolutional 2D Layer ---

class Conv2DLayer extends Layer {
    constructor(filters, kernelSize, stride=1, activation='relu', inputShape=null) {
        super();
        this.filters = filters; // Number of filters (output channels)
        this.kernelSize = kernelSize; // Kernel size: [height, width] (e.g., [3, 3])
        this.stride = stride; // Stride for convolution
        this.activationName = activation;
        this.activation = activations[activation];
        this.activationDerivative = activations[activation + 'Derivative'];
        this.kernels = null; // Convolution kernels (filters)
        this.biases = null;
        this.output = null;
        this.input = null;
        this.kernelsGradient = null;
        this.biasesGradient = null;

        if (inputShape) {
            this.build(inputShape);
        }
    }

    build(inputShape) { // inputShape: [height, width, channels]
        this.inputShape = inputShape;
        this.outputShape = this.calculateOutputShape();
        // Initialize kernels and biases
        const kernelScale = Math.sqrt(2.0 / (this.kernelSize[0] * this.kernelSize[1] * this.inputShape[2])); // He initialization adapted for Conv2D
        this.kernels = random([this.filters, this.kernelSize[0], this.kernelSize[1], this.inputShape[2]], kernelScale); // [filters, kernel_height, kernel_width, input_channels]
        this.biases = zeros([this.filters, 1]); // [filters, 1] - one bias per filter (as column vector)
    }

    calculateOutputShape() {
        const inputHeight = this.inputShape[0];
        const inputWidth = this.inputShape[1];
        const outputHeight = Math.floor((inputHeight - this.kernelSize[0]) / this.stride) + 1;
        const outputWidth = Math.floor((inputWidth - this.kernelSize[1]) / this.stride) + 1;
        return [outputHeight, outputWidth, this.filters]; // [output_height, output_width, filters]
    }

    forward(input) { // Optimized forward pass
        if (!this.kernels) {
            this.build(shape(input));
        }
        this.input = input;

        const inputHeight = this.inputShape[0];
        const inputWidth = this.inputShape[1];
        const inputChannels = this.inputShape[2];
        const outputHeight = this.outputShape[0];
        const outputWidth = this.outputShape[1];
        const kernelHeight = this.kernelSize[0];
        const kernelWidth = this.kernelSize[1];
        const stride = this.stride;
        const filters = this.filters;
        const kernels = this.kernels;
        const biases = this.biases;
        const activationFunc = this.activation;

        let output = zeros(this.outputShape); // Initialize output feature map

        for (let f = 0; f < filters; f++) { // Filter loop (outermost for potential data reuse)
            for (let y = 0; y < outputHeight; y++) {
                for (let x = 0; x < outputWidth; x++) {
                    let sum = 0;
                    for (let ky = 0; ky < kernelHeight; ky++) {
                        for (let kx = 0; kx < kernelWidth; kx++) {
                            for (let c = 0; c < inputChannels; c++) {
                                sum += input[y * stride + ky][x * stride + kx][c] * kernels[f][ky][kx][c];
                            }
                        }
                    }
                    output[y][x][f] = activationFunc(sum + biases[f][0]); // Activation applied directly here
                }
            }
        }
        this.output = output;
        return output;
    }

    backward(outputGradient) { // outputGradient shape: [output_height, output_width, filters] (from layer above)
        // 1. Derivative of activation
        const activationDerivativeVal = this.activationDerivative(this.output);
        const delta = multiply(outputGradient, activationDerivativeVal); // Shape: [output_height, output_width, filters]

        // Initialize gradients
        this.kernelsGradient = zeros(shape(this.kernels)); // Shape: [filters, kernel_height, kernel_width, input_channels]
        this.biasesGradient = zeros(shape(this.biases));     // Shape: [filters, 1]
        let inputGradient = zeros(shape(this.input));         // Shape: [input_height, input_width, input_channels]

        const inputHeight = this.inputShape[0];
        const inputWidth = this.inputShape[1];
        const inputChannels = this.inputShape[2];
        const outputHeight = this.outputShape[0];
        const outputWidth = this.outputShape[1];


        for (let f = 0; f < this.filters; f++) { // For each filter
            let bias_grad_sum = 0;
            for (let y = 0; y < outputHeight; y++) {
                for (let x = 0; x < outputWidth; x++) {
                    bias_grad_sum += delta[y][x][f]; // Summing up delta for bias gradient
                    for (let ky = 0; ky < this.kernelSize[0]; ky++) {
                        for (let kx = 0; kx < this.kernelSize[1]; kx++) {
                            for (let c = 0; c < inputChannels; c++) {
                                this.kernelsGradient[f][ky][kx][c] += input[y * this.stride + ky][x * this.stride + kx][c] * delta[y][x][f]; // Kernel gradient calculation
                                inputGradient[y * this.stride + ky][x * this.stride + kx][c] += this.kernels[f][ky][kx][c] * delta[y][x][f]; // Input gradient (for backpropagation) - WRONG - needs to be convolution of delta with rotated kernel
                            }
                        }
                    }
                }
            }
            this.biasesGradient[f][0] = bias_grad_sum; // Bias gradient for filter f
        }

        // Corrected Input Gradient Calculation (Convolution with Rotated Kernels and Stride) - (Simplified for stride=1 for now, stride>1 needs padding/upsampling in input gradient accumulation)
        inputGradient = zeros(shape(this.input)); // Re-initialize
        const paddedDelta = zeros([inputHeight, inputWidth, this.filters]); // Padding not needed if stride=1 and kernel <= input - but for general case, padding might be needed for accurate input gradient shape

        for (let f = 0; f < this.filters; f++) {
            for (let y = 0; y < outputHeight; y++) {
                for (let x = 0; x < outputWidth; x++) {
                    for (let ky = 0; ky < this.kernelSize[0]; ky++) {
                        for (let kx = 0; kx < this.kernelSize[1]; kx++) {
                            for (let c_in = 0; c_in < inputChannels; c_in++) { // Accumulate for each input channel
                                inputGradient[y * this.stride + ky][x * this.stride + kx][c_in] += this.kernels[f][this.kernelSize[0] - 1 - ky][this.kernelSize[1] - 1 - kx][c_in] * delta[y][x][f]; // Convolve delta with *rotated* kernel
                            }
                        }
                    }
                }
            }
        }


        return inputGradient;
    }


    getParameters() {
        return [this.kernels, this.biases];
    }

    getGradients() {
        return [this.kernelsGradient, this.biasesGradient];
    }

    applyGradients(learningRate) {
        this.kernels = subtract(this.kernels, scalarMultiply(this.kernelsGradient, learningRate));
        this.biases = subtract(this.biases, scalarMultiply(this.biasesGradient, learningRate));
    }
}

// --- Max Pooling 2D Layer ---

class MaxPooling2DLayer extends Layer {
    constructor(poolSize=[2, 2], stride=null) { // poolSize: [height, width], stride: defaults to poolSize if null
        super();
        this.poolSize = poolSize;
        this.stride = stride || poolSize; // Stride defaults to pool size if not specified
        this.output = null;
        this.input = null;
        this.outputIndices = null; // Store indices of max values for backprop
    }

    build(inputShape) { // inputShape: [height, width, channels]
        this.inputShape = inputShape;
        this.outputShape = this.calculateOutputShape();
    }

    calculateOutputShape() {
        const inputHeight = this.inputShape[0];
        const inputWidth = this.inputShape[1];
        const outputHeight = Math.floor((inputHeight - this.poolSize[0]) / this.stride[0]) + 1;
        const outputWidth = Math.floor((inputWidth - this.poolSize[1]) / this.stride[1]) + 1;
        return [outputHeight, outputWidth, this.inputShape[2]]; // Channels remain the same
    }


    forward(input) { // Input: [input_height, input_width, channels]
        if (!this.outputShape) {
            this.build(shape(input));
        }
        this.input = input;
        const inputHeight = this.inputShape[0];
        const inputWidth = this.inputShape[1];
        const inputChannels = this.inputShape[2];
        const outputHeight = this.outputShape[0];
        const outputWidth = this.outputShape[1];

        let output = zeros(this.outputShape);
        this.outputIndices = zeros(this.outputShape.concat(2)); // Store (y, x) indices of max values

        for (let c = 0; c < inputChannels; c++) { // For each channel
            for (let y = 0; y < outputHeight; y++) {
                for (let x = 0; x < outputWidth; x++) {
                    let maxVal = -Infinity;
                    let maxIndex = [0, 0];
                    for (let ky = 0; ky < this.poolSize[0]; ky++) {
                        for (let kx = 0; kx < this.poolSize[1]; kx++) {
                            const currentVal = input[y * this.stride[0] + ky][x * this.stride[1] + kx][c];
                            if (currentVal > maxVal) {
                                maxVal = currentVal;
                                maxIndex = [ky, kx]; // Relative index within the pool
                            }
                        }
                    }
                    output[y][x][c] = maxVal;
                    this.outputIndices[y][x][c] = maxIndex; // Store the index for backprop
                }
            }
        }
        this.output = output;
        return this.output;
    }


    backward(outputGradient) { // outputGradient: [output_height, output_width, channels]
        let inputGradient = zeros(this.inputShape); // Initialize input gradient
        const outputHeight = this.outputShape[0];
        const outputWidth = this.outputShape[1];
        const inputChannels = this.inputShape[2];


        for (let c = 0; c < inputChannels; c++) {
            for (let y = 0; y < outputHeight; y++) {
                for (let x = 0; x < outputWidth; x++) {
                    const maxIndex = this.outputIndices[y][x][c]; // Get stored max index
                    inputGradient[y * this.stride[0] + maxIndex[0]][x * this.stride[1] + maxIndex[1]][c] += outputGradient[y][x][c]; // Propagate gradient only to the max index location
                }
            }
        }
        return inputGradient;
    }
}


// --- Flatten Layer ---

class FlattenLayer extends Layer {
    constructor() {
        super();
        this.originalShape = null; // Store original shape for backprop
    }

    build(inputShape) { // inputShape: [height, width, channels] or any multi-dim tensor
        this.inputShape = inputShape;
        this.outputShape = [size(inputShape)]; // Output is a 1D vector
    }

    forward(input) {
        if (!this.outputShape) {
            this.build(shape(input));
        }
        this.originalShape = shape(input); // Store original shape for backward pass
        let flattened = [];
        traverseTensor(input, (val) => flattened.push(val));
        this.output = reshape(flattened, this.outputShape); // Reshape to 1D vector
        return this.output;
    }

    backward(outputGradient) { // outputGradient: [output_size, 1] - 1D vector gradient
        return reshape(outputGradient, this.originalShape); // Reshape back to original input shape
    }
}

// --- CNN (Convolutional Neural Network) Class ---

class CNN {
    constructor(layers=[]) { // Takes an array of layers
        this.layers = layers;
        this.inputShape = null; // Set when the first input is passed or defined explicitly
    }

    addLayer(layer) {
        this.layers.push(layer);
    }

    build(inputShape) { // To explicitly define the input shape and build layers
        this.inputShape = inputShape;
        let currentShape = inputShape;
        for (let layer of this.layers) {
            layer.build(currentShape);
            currentShape = layer.outputShape;
        }
    }

    forward(input) {
        if (!this.inputShape) { // Lazy build on first forward pass if not built explicitly
            this.build(shape(input));
        }
        let output = input;
        for (let layer of this.layers) {
            output = layer.forward(output);
        }
        return output;
    }

    backward(outputGradient) { // Backpropagate gradients through all layers (reverse order)
        let gradient = outputGradient;
        for (let i = this.layers.length - 1; i >= 0; i--) {
            gradient = this.layers[i].backward(gradient);
        }
        return gradient; // Returns the gradient w.r.t. the input of the *first* layer (not usually needed to be used directly)
    }

    getParameters() { // Get all trainable parameters from all layers
        let parameters = [];
        for (let layer of this.layers) {
            parameters = parameters.concat(layer.getParameters());
        }
        return parameters;
    }

    getGradients() { // Get all gradients from all layers
        let gradients = [];
        for (let layer of this.layers) {
            gradients = gradients.concat(layer.getGradients());
        }
        return gradients;
    }

    applyGradients(learningRate) { // Apply gradients to all layers
        for (let layer of this.layers) {
            layer.applyGradients(learningRate);
        }
    }

    summary() { // Print a summary of the network architecture
        console.log("--- CNN Summary ---");
        let inputShape = this.inputShape;
        console.log(`Input Shape: [${inputShape}]`);
        for (let layer of this.layers) {
            let layerType = layer.constructor.name;
            let outputShape = layer.outputShape;
            let paramCount = 0;
            layer.getParameters().forEach(paramTensor => paramCount += size(shape(paramTensor)));
            console.log(`${layerType} | Input Shape: [${inputShape}] | Output Shape: [${outputShape}] | Parameters: ${paramCount}`);
            inputShape = outputShape; // Update input shape for next layer in summary
        }
        console.log("--- End Summary ---");
    }
}

// --- DQN (Deep Q-Network) Class ---

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


// --- Replay Buffer Class ---

class ReplayBuffer {
    constructor(capacity) {
        this.capacity = capacity;
        this.buffer = [];
        this.position = 0;
    }

    add(state, action, reward, nextState, done) {
        if (this.buffer.length < this.capacity) {
            this.buffer.push(null); // Initialize if buffer not full yet
        }
        this.buffer[this.position] = { state, action, reward, nextState, done };
        this.position = (this.position + 1) % this.capacity; // Circular buffer
    }

    sample(batchSize) {
        const batch = {
            states: [],
            actions: [],
            rewards: [],
            nextStates: [],
            dones: []
        };
        const indices = Array.from({ length: this.size() }, (_, i) => i); // Indices array
        shuffleArray(indices); // Shuffle indices to get random samples
        const sampleIndices = indices.slice(0, batchSize); // Take batchSize random indices

        for (let index of sampleIndices) {
            const item = this.buffer[index];
            batch.states.push(item.state);
            batch.actions.push(item.action);
            batch.rewards.push(item.reward);
            batch.nextStates.push(item.nextState);
            batch.dones.push(item.done);
        }
        return batch;
    }

    size() {
        return this.buffer.length; // Returns current size, not capacity (important for initialization phase)
    }
}


// --- Loss Functions ---

function meanSquaredError(predicted, target) {
    const diff = subtract(predicted, target);
    const squaredDiff = multiply(diff, diff);
    return mean(squaredDiff);
}

function meanSquaredErrorDerivative(predicted, target) { // Derivative of MSE w.r.t. 'predicted'
    return scalarMultiply(scalarMultiply(subtract(predicted, target), 2.0), 1.0/size(shape(predicted))); // 2 * (predicted - target) / N
}


// --- Utility Functions for DQN ---

function argMax(tensor) { // Returns index of maximum value in a 1D tensor
    if (!Array.isArray(tensor) || shape(tensor).length !== 1) {
        throw new Error("argMax function expects a 1D tensor.");
    }
    let maxVal = -Infinity;
    let maxIndex = -1;
    for (let i = 0; i < tensor.length; i++) {
        if (tensor[i] > maxVal) {
            maxVal = tensor[i];
            maxIndex = i;
        }
    }
    return maxIndex;
}

function cloneNetwork(network) { // Deep clone a CNN network (for target network)
    const clonedLayers = [];
    for (let layer of network.layers) {
        let clonedLayer;
        if (layer instanceof Conv2DLayer) {
            clonedLayer = new Conv2DLayer(layer.filters, layer.kernelSize, layer.stride, layer.activationName);
            clonedLayer.kernels = clone(layer.kernels);
            clonedLayer.biases = clone(layer.biases);
            clonedLayer.build(layer.inputShape); // Ensure outputShape etc are calculated
        } else if (layer instanceof MaxPooling2DLayer) {
            clonedLayer = new MaxPooling2DLayer(layer.poolSize, layer.stride);
            clonedLayer.build(layer.inputShape);
        } else if (layer instanceof FlattenLayer) {
            clonedLayer = new FlattenLayer();
            clonedLayer.build(layer.inputShape);
        } else if (layer instanceof DenseLayer) {
            clonedLayer = new DenseLayer(layer.units, layer.activationName);
            clonedLayer.weights = clone(layer.weights);
            clonedLayer.biases = clone(layer.biases);
            clonedLayer.build(layer.inputShape);
        } else {
            throw new Error("Unsupported layer type for cloning.");
        }
        clonedLayers.push(clonedLayer);
    }
    return new CNN(clonedLayers);
}


function shuffleArray(array) { // In-place Fisher-Yates shuffle for replay buffer sampling
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]]; // Swap elements
    }
}

// --- Conceptual Game Loop (Illustrative - you need to adapt to your game) ---

async function gameLoop() {
    const stateShape = [84, 84, 4]; // Example state shape (adjust to your game)
    const actionSpaceSize = 4; // Example action space size (adjust to your game)

    const dqnAgent = new DQN(stateShape, actionSpaceSize);
    dqnAgent.summary(); // Print DQN architecture summary

    const numEpisodes = 10000; // Number of training episodes
    const maxStepsPerEpisode = 1000; // Maximum steps per episode

    for (let episode = 0; episode < numEpisodes; episode++) {
        let currentState = resetGameEnvironment(); // Function to reset your game environment, returns initial state
        let episodeReward = 0;

        for (let step = 0; step < maxStepsPerEpisode; step++) {
            // 1. Agent chooses action
            const action = dqnAgent.getAction(currentState);

            // 2. Environment step (you need to implement your game environment interaction)
            const { nextState, reward, done } = gameEnvironmentStep(currentState, action); // Returns next state, reward, and done flag

            episodeReward += reward;

            // 3. Store transition in replay buffer
            dqnAgent.storeTransition(currentState, action, reward, nextState, done);

            // 4. Train DQN agent (if replay buffer has enough samples)
            const loss = dqnAgent.train();
            if (loss) { // Training happens only if replay buffer is filled enough
                // console.log(`Episode: ${episode}, Step: ${step}, Loss: ${loss.toFixed(4)}, Exploration Rate: ${dqnAgent.explorationRate.toFixed(3)}`);
            }

            currentState = nextState; // Move to next state

            if (done) {
                break; // Episode finished
            }
            // await new Promise(resolve => setTimeout(resolve, 0)); // Optional delay for visualization if needed, use with caution
        }
        console.log(`Episode: ${episode}, Total Reward: ${episodeReward}, Exploration Rate: ${dqnAgent.explorationRate.toFixed(3)}`);

        // Periodically save weights (optional)
        if (episode % 100 === 0) {
            // saveDQNWeights(dqnAgent, `dqn_weights_episode_${episode}.json`); // Function to save weights
        }
    }
    console.log("Training finished!");
}

// --- Tetris Game Environment Integration ---

function resetGameEnvironment() {
    console.log("resetGameEnvironment() called - Creating new Tetris game instance"); // Log reset
    const tetrisGame = new Tetris(); // Create a new Tetris game instance
    tetrisGame.gameNumber++; // Increment game number for each new game
    return getTetrisGameStateTensor(tetrisGame); // Get initial game state tensor
}

function gameEnvironmentStep(stateTensor, actionIndex) {
    console.log(`gameEnvironmentStep() called - Action Index: ${actionIndex}`); // Log action

    // --- Action Mapping (Map action index to Tetris actions) ---
    const actions = ['left', 'right', 'down', 'rotate', '']; // '' is no-op, or you can adjust action space
    const action = actions[actionIndex]; // Map action index to action name
    console.log(`Mapped Action: ${action}`);

    // --- Reconstruct Tetris Game State from Tensor (if needed, based on how you represent state) ---
    // For now, assuming stateTensor is not directly used to reconstruct game state for action,
    // but if your state representation requires it, you'd need to reverse the tensor conversion here.
    // In this example, we're directly controlling the *current* Tetris game instance.

    // --- Get the current Tetris game instance (you might need to manage this globally or differently) ---
    // Assuming 'tetrisGame' is accessible in this scope (you might need to adjust based on your setup)
    if (!tetrisGame) {
        console.error("Error: tetrisGame instance is not available in gameEnvironmentStep(). Make sure it's properly initialized and accessible.");
        return { nextState: zeros([84, 84, 4]), reward: -10, done: true }; // Return error state
    }


    // --- Perform Tetris Game Step ---
    tetrisGame.clearTetromino(); // Clear active tetromino before applying action (important for move/rotate to work correctly)
    tetrisGame.step(action);     // Perform Tetris game step with the chosen action
    tetrisGame.placeTetromino(); // Re-place the tetromino after action and gravity


    // --- Calculate Reward (you'll need to design your reward function for Tetris) ---
    let reward = 1; // Example reward: +1 for surviving a step - REPLACE with your reward logic!
    if (tetrisGame.score > 0) {
        reward += tetrisGame.score; // Example: reward proportional to score increase
        tetrisGame.score = 0; // Reset score in Tetris game instance after giving reward (or adjust as needed)
    }
    if (tetrisGame.gameOver) {
        reward = -10; // Example: Negative reward for game over - ADJUST!
    }


    // --- Get Next State as Tensor ---
    const nextStateTensor = getTetrisGameStateTensor(tetrisGame);

    // --- Determine 'done' flag ---
    const done = tetrisGame.gameOver;
    console.log(`Step Result - Reward: ${reward}, Done: ${done}, Game Over: ${tetrisGame.gameOver}`);


    return { nextState: nextStateTensor, reward: reward, done: done };
}


function getTetrisGameStateTensor(tetrisGame) {
    // --- Convert Tetris game board to a tensor (e.g., 84x84x4) ---
    // This is a placeholder - you'll need to implement the actual conversion
    // based on how you want to represent the Tetris state to your DQN.

    console.log("getTetrisGameStateTensor() called - Converting Tetris board to tensor - PLACEHOLDER IMPLEMENTATION!");

    const height = tetrisGame.height;
    const width = tetrisGame.width;
    const board = tetrisGame.board;

    // --- Example: Very basic grayscale representation (you'll likely want something more sophisticated) ---
    const tensorState = zeros([84, 84, 4]); // Initialize tensor with zeros (example shape)

    // --- Simple board to tensor conversion (example - adapt as needed) ---
    for (let y = 0; y < Math.min(height, 84); y++) { // Scale Tetris board to 84x84 (or use padding, resizing etc.)
        for (let x = 0; x < Math.min(width, 84); x++) {
            if (y < height && x < width) {
                if (board[y][x] === 1 || board[y][x] === 2) { // If cell is occupied (1 or 2), set to white (or some value)
                    tensorState[y][x][0] = 255; // Example: Grayscale value 255 for occupied cell (you might normalize to 0-1)
                    tensorState[y][x][1] = 255;
                    tensorState[y][x][2] = 255;
                    tensorState[y][x][3] = 255; // Alpha channel (if used)
                } else {
                    // Cell is empty (0), keep as black (or your background color) - tensorState already initialized to zeros
                }
            }
        }
    }

    console.log(`Tensor State Shape: ${shape(tensorState)}`); // Log tensor state shape
    return tensorState;
}


// ---  Global Tetris Game Instance (Important: You might need to manage this differently based on your game loop structure) ---
let tetrisGame; // Global variable to hold the current Tetris game instance
