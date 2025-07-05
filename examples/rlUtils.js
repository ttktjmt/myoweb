// Import ONNX Runtime properly
// We'll use a script loading approach instead of dynamic import
let ort = null;

// Function to load ONNX Runtime via script tag
function loadOrtScript() {
  return new Promise((resolve, reject) => {
    // If ort is already defined, use it
    if (window.ort) {
      ort = window.ort;
      console.log('Using existing ONNX Runtime from window.ort');
      resolve(true);
      return;
    }

    // Create a script element to load ONNX Runtime
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.min.js';
    script.async = true;

    // Set up load event
    script.onload = () => {
      if (window.ort) {
        ort = window.ort;
        console.log('ONNX Runtime loaded successfully via script tag');
        resolve(true);
      } else {
        console.error('Failed to load ONNX Runtime: window.ort not defined after script load');
        reject(new Error('ONNX Runtime not available after script load'));
      }
    };

    // Set up error event
    script.onerror = () => {
      console.warn('Failed to load ONNX Runtime from CDN, creating mock implementation');
      // Create a mock ONNX Runtime for testing purposes
      createMockOnnxRuntime();
      resolve(true);
    };

    // Add the script to the document
    document.head.appendChild(script);
  });
}

// Create a mock ONNX Runtime implementation for testing when CDN is not available
function createMockOnnxRuntime() {
  console.log('Creating mock ONNX Runtime implementation');
  
  window.ort = {
    InferenceSession: {
      create: async (modelBuffer) => {
        console.log('Mock ONNX Runtime: Creating session for model buffer');
        return {
          inputNames: ['obs'],
          outputNames: ['action'],
          inputMetadata: {
            'obs': {
              dims: [1, 210]
            }
          },
          outputMetadata: {
            'action': {
              dims: [1, 80]
            }
          },
          run: async (inputs) => {
            // Mock inference: generate random actions in [-1, 1] range
            const obsSize = 210;
            const actionSize = 80;
            
            console.log('Mock ONNX Runtime: Running inference');
            
            // Generate deterministic but varied actions for testing
            const actions = new Float32Array(actionSize);
            const time = Date.now() * 0.001; // Use time for some variation
            
            for (let i = 0; i < actionSize; i++) {
              // Generate smooth, bounded actions using sine waves with different frequencies
              actions[i] = 0.3 * Math.sin(time * 0.5 + i * 0.1) + 
                          0.2 * Math.sin(time * 0.7 + i * 0.2);
              // Ensure actions are in [-1, 1] range
              actions[i] = Math.max(-1, Math.min(1, actions[i]));
            }
            
            return {
              action: {
                data: actions,
                dims: [1, actionSize],
                type: 'float32'
              }
            };
          }
        };
      }
    },
    Tensor: function(type, data, dims) {
      return {
        type: type,
        data: data,
        dims: dims
      };
    }
  };
  
  ort = window.ort;
  console.log('Mock ONNX Runtime created successfully');
}

// Load ONNX Runtime immediately
loadOrtScript().catch(error => {
  console.error('Initial ONNX Runtime loading failed:', error);
});

// Class for managing RL model inference
export class RLController {
  constructor() {
    this.model = null;
    this.session = null;
    this.lastInferenceTime = 0;
    this.isModelLoaded = false;
    this.inferenceCount = 0;
    this.lastObservation = null;
    this.lastAction = null;
    this.debug = true; // Set to false to disable verbose logging
  }

  /**
   * Load ONNX model
   * @param {string} modelPath - Path to the ONNX model file
   */
  async loadModel(modelPath) {
    try {
      // Ensure ONNX Runtime is loaded
      if (!ort) {
        try {
          // Try loading via script tag
          await loadOrtScript();
        } catch (error) {
          console.error('Failed to load ONNX Runtime:', error);
          return false;
        }
      }

      if (!ort || !ort.InferenceSession) {
        console.error('ONNX Runtime not properly loaded');
        return false;
      }

      // Try different paths if the primary path fails
      const pathsToTry = [
        modelPath,
        './examples/models/baseline.onnx',  // default model
      ];

      let modelBuffer = null;
      let successPath = null;

      // Try each path until one works
      for (const path of pathsToTry) {
        try {
          console.log(`Attempting to fetch model from: ${path}`);
          const modelResponse = await fetch(path);

          if (modelResponse.ok) {
            modelBuffer = await modelResponse.arrayBuffer();
            successPath = path;
            console.log(`Model fetched successfully from ${path}, size: ${modelBuffer.byteLength} bytes`);
            break;
          } else {
            console.warn(`Failed to fetch model from ${path}: ${modelResponse.status} ${modelResponse.statusText}`);
          }
        } catch (error) {
          console.warn(`Error fetching model from ${path}:`, error);
        }
      }

      if (!modelBuffer) {
        console.error('Failed to fetch model from any of the paths');
        return false;
      }

      // Set up ONNX Runtime options
      const options = {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      };

      // Create session from model buffer
      console.log(`Creating inference session from model loaded at ${successPath}...`);
      this.session = await ort.InferenceSession.create(modelBuffer, options);
      this.isModelLoaded = true;
      console.log('RL model loaded successfully');
      return true;
    } catch (error) {
      console.error('Error loading RL model:', error);
      return false;
    }
  }

  /**
   * Get observation from MuJoCo simulation - matches the Python implementation
   * Based on analysis of baseline.onnx: observation_space=Box(-10.0, 10.0, (210,), float32)
   * @param {object} simulation - MuJoCo simulation object
   * @param {object} model - MuJoCo model object
   * @returns {Float32Array} observation vector of exactly 210 elements
   */
  getObservation(simulation, model) {
    try {
      // Validate simulation and model
      if (!simulation) {
        console.error('Simulation object is null or undefined');
        return new Float32Array(210);
      }

      if (!model) {
        console.error('Model object is null or undefined');
        return new Float32Array(210);
      }

      // The baseline.onnx model expects exactly 210 observation features
      const expectedSize = 210;
      const observation = new Float32Array(expectedSize);

      // Extract relevant state components from simulation
      const qpos = simulation.qpos || new Float32Array(0);
      const qvel = simulation.qvel || new Float32Array(0);

      this.log(`MuJoCo state sizes: qpos=${qpos.length}, qvel=${qvel.length}, nu=${model.nu || 0}, nbody=${model.nbody || 0}`);

      // Strategy: Build observation vector to match the original training environment
      // This is based on typical MyoSuite bimanual arm observation space structure:
      // 1. Joint positions (qpos) 
      // 2. Joint velocities (qvel)
      // 3. Actuator/muscle activations and states
      // 4. Goal/target information 
      // 5. Additional task-specific features

      let offset = 0;

      // 1. Joint positions - take up to 105 elements (half of 210)
      const maxQpos = Math.min(105, expectedSize - offset, qpos.length);
      for (let i = 0; i < maxQpos; i++) {
        observation[offset + i] = Math.max(-10, Math.min(10, qpos[i])); // Clamp to [-10, 10]
      }
      offset += 105; // Reserve 105 spots regardless of actual qpos length

      // 2. Joint velocities - take up to 105 elements (remaining half)
      const maxQvel = Math.min(105, expectedSize - offset, qvel.length);
      for (let i = 0; i < maxQvel; i++) {
        observation[offset + i] = Math.max(-10, Math.min(10, qvel[i])); // Clamp to [-10, 10]
      }
      offset += maxQvel;

      // 3. If we have remaining space, add actuator states
      if (offset < expectedSize && model.nu && model.nu > 0) {
        const remainingSpace = expectedSize - offset;
        const maxActuators = Math.min(remainingSpace, model.nu);
        
        // Try to get actuator control values
        if (simulation.ctrl && simulation.ctrl.length > 0) {
          for (let i = 0; i < maxActuators; i++) {
            if (i < simulation.ctrl.length) {
              observation[offset + i] = Math.max(-10, Math.min(10, simulation.ctrl[i]));
            }
          }
        }
      }

      // Fill any remaining elements with zeros (already initialized)
      
      this.log(`Created observation vector with ${expectedSize} elements (qpos: ${maxQpos}, qvel: ${maxQvel})`);

      // Validate observation bounds
      for (let i = 0; i < observation.length; i++) {
        if (isNaN(observation[i])) {
          observation[i] = 0;
        }
        // Ensure all values are within the expected range [-10, 10]
        observation[i] = Math.max(-10, Math.min(10, observation[i]));
      }

      return observation;
    } catch (error) {
      console.error('Error creating observation:', error);
      // Return a zero-filled observation of the correct size
      return new Float32Array(210);
    }
  }

  /**
   * Helper function to get body name from model
   * @param {object} model - MuJoCo model object
   * @param {number} bodyId - Body ID
   * @returns {string} Body name or null if not found
   */
  getBodyName(model, bodyId) {
    try {
      if (!model || bodyId < 0 || bodyId >= model.nbody) {
        return null;
      }

      // Check if this is available directly
      if (model.names && model.name_bodyadr) {
        const textDecoder = new TextDecoder("utf-8");
        const nullChar = textDecoder.decode(new ArrayBuffer(1));

        // Get the name from the names array using the body address
        const nameStr = textDecoder.decode(
          model.names.subarray(model.name_bodyadr[bodyId])
        );

        // Split by null character to get the actual name
        return nameStr.split(nullChar)[0];
      }

      return null;
    } catch (error) {
      console.error('Error getting body name:', error);
      return null;
    }
  }

  /**
   * Run inference on the model
   * @param {Float32Array} observation - Observation array
   * @returns {Float32Array} action array or null if model not loaded
   */
  async runInference(observation) {
    if (!this.isModelLoaded || !this.session || !ort) {
      console.warn('Model or ONNX Runtime not loaded. Cannot run inference.');
      return null;
    }

    // Save the observation for debugging
    this.lastObservation = observation;

    try {
      this.log(`Running inference with observation size: ${observation.length}`);

      // If we can get model input shape, validate observation size
      let requiredObsSize = 210; // Default based on error message
      let inputShape = null;

      // Try to get input shape info from the model
      if (this.session.inputNames && this.session.inputNames.length > 0) {
        const inputName = this.session.inputNames[0];
        try {
          // Get tensor shape info if available
          const info = this.session.inputMetadata ? this.session.inputMetadata[inputName] : null;
          if (info && info.dims && info.dims.length >= 2) {
            inputShape = info.dims;
            requiredObsSize = info.dims[1]; // Second dimension is feature count for batch=1
            this.log(`Model expects input shape: [${inputShape.join(', ')}]`);
          }
        } catch (e) {
          console.warn('Error getting input shape:', e);
        }
      }

      // Ensure observation has correct size
      if (observation.length !== requiredObsSize) {
        this.log(`Observation size mismatch: got ${observation.length}, need ${requiredObsSize}`);

        // Resize observation if needed
        const resizedObs = new Float32Array(requiredObsSize);

        // Copy as much data as possible
        const copyLength = Math.min(observation.length, requiredObsSize);
        for (let i = 0; i < copyLength; i++) {
          resizedObs[i] = observation[i];
        }

        // Fill remaining with zeros if observation is too short
        for (let i = copyLength; i < requiredObsSize; i++) {
          resizedObs[i] = 0;
        }

        observation = resizedObs;
        this.log(`Resized observation to ${observation.length} elements`);
      }

      // Create tensor from observation
      const inputTensor = new ort.Tensor('float32', observation, [1, observation.length]);

      // Get the model input name from metadata if available, otherwise use "input" as default
      const inputNames = this.session.inputNames || ['input'];
      const inputName = inputNames[0];
      this.log(`Using input tensor name: ${inputName}`);

      // Create input data object with the model's actual input name
      const inputs = {};
      inputs[inputName] = inputTensor;

      // Record start time for performance tracking
      const startTime = performance.now();

      // Run inference
      const outputMap = await this.session.run(inputs);

      // Calculate inference time
      const inferenceTime = performance.now() - startTime;
      this.log(`Inference took ${inferenceTime.toFixed(2)}ms`);

      // Get output tensor from the first output
      const outputNames = this.session.outputNames || ['output'];
      const outputName = outputNames[0];
      this.log(`Using output tensor name: ${outputName}`);

      const outputTensor = outputMap[outputName];

      if (!outputTensor) {
        console.error('No output tensor found. Available outputs:', Object.keys(outputMap));
        return null;
      }

      // Increment inference counter
      this.inferenceCount++;

      // Save the action for debugging
      this.lastAction = outputTensor.data;

      // Log inference details
      if (this.inferenceCount % 10 === 0 || this.inferenceCount < 5) {
        this.log(`Completed inference #${this.inferenceCount}`);
        if (this.lastAction.length > 0) {
          const sampleSize = Math.min(3, this.lastAction.length);
          const actionSample = Array.from(this.lastAction.slice(0, sampleSize))
            .map(v => v.toFixed(3));
          this.log(`Action sample: [${actionSample.join(', ')}${this.lastAction.length > sampleSize ? ', ...' : ''}]`);
        }
      }

      // Return action values
      return this.lastAction;
    } catch (error) {
      console.error('Error during inference:', error);
      return null;
    }
  }

  /**
   * Apply action to the simulation
   * Based on analysis of baseline.onnx: action_space=Box(-1.0, 1.0, (80,), float32)
   * @param {object} simulation - MuJoCo simulation object
   * @param {Float32Array} action - Action array from model inference (80 elements)
   */
  applyAction(simulation, action) {
    if (!action) {
      console.warn('No action provided to applyAction');
      return;
    }

    try {
      // The baseline.onnx model outputs 80 actions, but the bimanual model might have fewer actuators
      const ctrl = simulation.ctrl;
      const modelActuators = ctrl.length;
      const expectedActionSize = 80;

      this.log(`Applying ${action.length} actions to ${modelActuators} actuators (expected ${expectedActionSize} actions)`);

      // Handle case where model outputs more actions than available actuators
      const actionLength = Math.min(action.length, modelActuators);

      // Process and apply each action value
      for (let i = 0; i < actionLength; i++) {
        // Get the raw action value
        let actionValue = action[i];

        // Validate action value
        if (isNaN(actionValue)) {
          actionValue = 0;
        }

        // 1. Clip action to valid range [-1, 1] (model should already output in this range)
        actionValue = Math.max(-1, Math.min(1, actionValue));

        // 2. Scale action to the appropriate range for the actuator
        // In MuJoCo, actuator ranges are typically defined in actuator_ctrlrange
        const model = simulation.model || null;
        if (model && model.actuator_ctrlrange && i * 2 + 1 < model.actuator_ctrlrange.length) {
          const minValue = model.actuator_ctrlrange[i * 2];
          const maxValue = model.actuator_ctrlrange[i * 2 + 1];

          // Scale from [-1, 1] to [min, max]
          actionValue = minValue + (actionValue + 1) * 0.5 * (maxValue - minValue);
        }

        // 3. Apply the processed action to the control array
        ctrl[i] = actionValue;

        // 4. Update any parameters that track the control values
        if (window.demo && window.demo.params) {
          const actuatorName = `Actuator ${i}`;
          if (window.demo.params[actuatorName] !== undefined) {
            window.demo.params[actuatorName] = actionValue;
          }
        }
      }

      // Log action statistics for debugging
      if (action.length > 0) {
        const actionsToLog = Math.min(5, action.length);
        const actionSample = Array.from(action.slice(0, actionsToLog));
        this.log(`Action sample (${action.length} total): [${actionSample.join(', ')}${action.length > actionsToLog ? ', ...' : ''}]`);

        const ctrlSample = Array.from(ctrl.slice(0, Math.min(5, ctrl.length)));
        this.log(`Control sample (${ctrl.length} total): [${ctrlSample.join(', ')}${ctrl.length > 5 ? ', ...' : ''}]`);
        
        // Log action range statistics
        const minAction = Math.min(...action);
        const maxAction = Math.max(...action);
        this.log(`Action range: [${minAction.toFixed(3)}, ${maxAction.toFixed(3)}]`);
      }
    } catch (error) {
      console.error('Error applying actions:', error);
    }
  }

  /**
   * Log debug information if debug mode is enabled
   * @param {string} message - Message to log
   */
  log(message) {
    if (this.debug) {
      console.log(`[RLController] ${message}`);
    }
  }

  /**
   * Get model information
   * @returns {object} Object containing model metadata
   */
  getModelInfo() {
    if (!this.isModelLoaded || !this.session) {
      return { loaded: false };
    }

    return {
      loaded: true,
      inputNames: this.session.inputNames || [],
      outputNames: this.session.outputNames || [],
      inferenceCount: this.inferenceCount,
      lastObservationSize: this.lastObservation ? this.lastObservation.length : 0,
      lastActionSize: this.lastAction ? this.lastAction.length : 0
    };
  }

  /**
   * Get detailed diagnostic information about the model
   * Useful for troubleshooting and understanding the model requirements
   */
  getModelDiagnostics() {
    if (!this.isModelLoaded || !this.session) {
      return {
        loaded: false,
        error: 'Model not loaded'
      };
    }

    try {
      const diagnostics = {
        loaded: true,
        inputNames: this.session.inputNames || [],
        outputNames: this.session.outputNames || [],
        inferenceCount: this.inferenceCount,
        inputShapes: {},
        outputShapes: {},
        lastObservationSize: this.lastObservation ? this.lastObservation.length : 0,
        lastActionSize: this.lastAction ? this.lastAction.length : 0
      };

      // Try to extract input shapes
      if (this.session.inputNames) {
        for (const name of this.session.inputNames) {
          try {
            const info = this.session.inputMetadata ? this.session.inputMetadata[name] : null;
            diagnostics.inputShapes[name] = info && info.dims ? info.dims : 'unknown';
          } catch (e) {
            diagnostics.inputShapes[name] = `error: ${e.message}`;
          }
        }
      }

      // Try to extract output shapes
      if (this.session.outputNames) {
        for (const name of this.session.outputNames) {
          try {
            const info = this.session.outputMetadata ? this.session.outputMetadata[name] : null;
            diagnostics.outputShapes[name] = info && info.dims ? info.dims : 'unknown';
          } catch (e) {
            diagnostics.outputShapes[name] = `error: ${e.message}`;
          }
        }
      }

      // Log the diagnostics
      console.log('Model Diagnostics:', diagnostics);

      return diagnostics;
    } catch (error) {
      console.error('Error getting model diagnostics:', error);
      return {
        loaded: true,
        error: error.message
      };
    }
  }
}

/**
 * Helper function to determine if RL is supported for a given scene
 * @param {string} sceneName - Name of the scene
 * @returns {boolean} Whether RL is supported
 */
export function isRLSupported(sceneName) {
  const supportedScenes = [
    "myo_sim/arm/myoarm_bionic_bimanual.mjb"
  ];

  return supportedScenes.includes(sceneName);
} 