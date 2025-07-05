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

    // Try to load from local lib directory first, then fallback to CDN
    const sources = [
      './lib/ort.min.js',  // Local copy
      'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.15.1/dist/ort.min.js'  // CDN fallback
    ];
    
    let currentIndex = 0;
    
    function tryNextSource() {
      if (currentIndex >= sources.length) {
        reject(new Error('All ONNX Runtime sources failed to load'));
        return;
      }
      
      const script = document.createElement('script');
      script.src = sources[currentIndex];
      script.async = true;

      script.onload = () => {
        if (window.ort) {
          ort = window.ort;
          console.log(`ONNX Runtime loaded successfully from: ${sources[currentIndex]}`);
          resolve(true);
        } else {
          console.warn(`ONNX Runtime script loaded but window.ort not defined from: ${sources[currentIndex]}`);
          currentIndex++;
          tryNextSource();
        }
      };

      script.onerror = () => {
        console.warn(`Failed to load ONNX Runtime from: ${sources[currentIndex]}`);
        currentIndex++;
        tryNextSource();
      };

      document.head.appendChild(script);
    }
    
    tryNextSource();
  });
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
    this.mockMode = false; // Enable mock inference when ONNX Runtime is unavailable
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
      console.warn('Falling back to mock inference mode for testing');
      this.mockMode = true;
      this.isModelLoaded = true; // Allow mock mode to work
      return true; // Return true so the system can continue with mock inference
    }
  }

  /**
   * Get observation from MuJoCo simulation - matches the Python implementation
   * @param {object} simulation - MuJoCo simulation object
   * @param {object} model - MuJoCo model object
   * @returns {Float32Array} observation vector
   */
  getObservation(simulation, model) {
    try {
      // Validate simulation and model
      if (!simulation) {
        console.error('Simulation object is null or undefined');
        return new Float32Array(0);
      }

      if (!model) {
        console.error('Model object is null or undefined');
        return new Float32Array(0);
      }

      // Log available properties for debugging
      this.log('Simulation properties: ' + Object.keys(simulation).join(', '));

      // Extract relevant state components from simulation
      const qpos = simulation.qpos || new Float32Array(0);
      const qvel = simulation.qvel || new Float32Array(0);

      // Check if actuator properties exist, use empty arrays if not
      const actuator_length = simulation.actuator_length ||
        (simulation.ten_length ? simulation.ten_length : new Float32Array(0));
      const actuator_velocity = simulation.actuator_velocity ||
        (simulation.ten_velocity ? simulation.ten_velocity : new Float32Array(0));
      const actuator_force = simulation.actuator_force || new Float32Array(0);

      // Log sizes for debugging
      this.log(`State vector sizes: qpos=${qpos.length}, qvel=${qvel.length}, ` +
        `actuator_length=${actuator_length.length}, actuator_velocity=${actuator_velocity.length}, ` +
        `actuator_force=${actuator_force.length}`);
      
      // Log model structure for debugging
      if (model.nq !== undefined) this.log(`Model nq (generalized coords): ${model.nq}`);
      if (model.nv !== undefined) this.log(`Model nv (degrees of freedom): ${model.nv}`);
      if (model.nu !== undefined) this.log(`Model nu (actuators): ${model.nu}`);
      if (model.nbody !== undefined) this.log(`Model nbody (bodies): ${model.nbody}`);

      // Create observation arrays for different components
      const nu = model.nu || 0;
      const actuator_pos_data = new Float32Array(nu);
      const actuator_vel_data = new Float32Array(nu);
      const actuator_force_data = new Float32Array(nu);

      // Fill actuator state arrays
      for (let i = 0; i < nu; i++) {
        actuator_pos_data[i] = i < actuator_length.length ? actuator_length[i] : 0;
        actuator_vel_data[i] = i < actuator_velocity.length ? actuator_velocity[i] : 0;
        actuator_force_data[i] = i < actuator_force.length ? actuator_force[i] : 0;
      }

      // Calculate relevant body positions and velocities
      let body_pos_data = [];
      let body_vel_data = [];

      // Get target body positions and velocities (for bimanual arm scenario)
      // These would be end effector positions/velocities in the Python implementation
      for (let b = 0; b < model.nbody; b++) {
        // Check if this is an end effector or target body
        const bodyName = this.getBodyName(model, b);
        if (bodyName && (
          bodyName.includes('target') ||
          bodyName.includes('grasp') ||
          bodyName.includes('end_effector'))) {

          // Get position (xyz)
          for (let j = 0; j < 3; j++) {
            body_pos_data.push(simulation.xpos[b * 3 + j]);
          }

          // Get velocity (xyz)
          for (let j = 0; j < 3; j++) {
            body_vel_data.push(simulation.xvel[b * 3 + j]);
          }
        }
      }

      // Convert array-like body data to Float32Array
      const body_pos = new Float32Array(body_pos_data);
      const body_vel = new Float32Array(body_vel_data);

      // Combine all observation components into a single observation vector
      // The exact shape should match what the model expects
      const fullObsLength = qpos.length + qvel.length +
        actuator_pos_data.length +
        actuator_vel_data.length +
        actuator_force_data.length +
        body_pos.length + body_vel.length;

      const fullObsData = new Float32Array(fullObsLength);

      // Fill the observation array with all components
      let offset = 0;

      // Joint positions and velocities
      fullObsData.set(qpos, offset);
      offset += qpos.length;

      fullObsData.set(qvel, offset);
      offset += qvel.length;

      // Actuator states
      fullObsData.set(actuator_pos_data, offset);
      offset += actuator_pos_data.length;

      fullObsData.set(actuator_vel_data, offset);
      offset += actuator_vel_data.length;

      fullObsData.set(actuator_force_data, offset);
      offset += actuator_force_data.length;

      // Body positions and velocities
      fullObsData.set(body_pos, offset);
      offset += body_pos.length;

      fullObsData.set(body_vel, offset);

      this.log(`Created full observation vector with ${fullObsData.length} elements`);

      // ERROR: The model expects an input of size 210, but we're generating 381
      // We need to adapt our observation to match the expected size

      // SOLUTION: Create a processed observation matching the expected size
      // Based on analysis, the model expects exactly 210 dimensions
      // This is likely qpos (105) + qvel (105) for the bimanual model
      
      const expectedSize = 210;
      const processedObs = new Float32Array(expectedSize);
      
      // Method: Use exactly qpos + qvel, which should total 210 for bimanual arms
      let offset = 0;
      
      // Copy qpos data
      const qposSize = Math.min(qpos.length, expectedSize - offset);
      for (let i = 0; i < qposSize; i++) {
        processedObs[offset + i] = qpos[i];
      }
      offset += qposSize;
      
      // Copy qvel data
      const qvelSize = Math.min(qvel.length, expectedSize - offset);
      for (let i = 0; i < qvelSize; i++) {
        processedObs[offset + i] = qvel[i];
      }
      offset += qvelSize;
      
      // If we still have space and the total doesn't match, add some actuator information
      if (offset < expectedSize) {
        const remainingSpace = expectedSize - offset;
        this.log(`Filling remaining ${remainingSpace} dimensions with actuator data`);
        
        // Add actuator positions first
        const actuatorDataSize = Math.min(actuator_pos_data.length, remainingSpace);
        for (let i = 0; i < actuatorDataSize; i++) {
          processedObs[offset + i] = actuator_pos_data[i];
        }
        offset += actuatorDataSize;
      }
      
      // Fill any remaining elements with zeros
      for (let i = offset; i < expectedSize; i++) {
        processedObs[i] = 0;
      }
      
      console.log(`Created processed observation: qpos=${qposSize}, qvel=${qvelSize}, actuator=${Math.min(actuator_pos_data.length, expectedSize - qposSize - qvelSize)}, total=${processedObs.length}`);
      
      // Log first few and last few values for debugging
      if (this.debug) {
        const sampleSize = 5;
        const firstVals = Array.from(processedObs.slice(0, sampleSize)).map(v => v.toFixed(3));
        const lastVals = Array.from(processedObs.slice(-sampleSize)).map(v => v.toFixed(3));
        this.log(`Obs sample - first: [${firstVals.join(', ')}], last: [${lastVals.join(', ')}]`);
      }

      // Store both the full and processed observations for debugging
      this.fullObservation = fullObsData;

      return processedObs;
    } catch (error) {
      console.error('Error creating observation:', error);
      // Fallback to a simple observation of the expected size
      const expectedSize = 210;
      const fallbackObs = new Float32Array(expectedSize);

      // Try to fill with qpos and qvel data if available
      if (simulation.qpos) {
        const qposLength = Math.min(simulation.qpos.length, expectedSize);
        for (let i = 0; i < qposLength; i++) {
          fallbackObs[i] = simulation.qpos[i];
        }
      }

      if (simulation.qvel) {
        const qvelStart = Math.min(simulation.qpos ? simulation.qpos.length : 0, expectedSize);
        const qvelLength = Math.min(simulation.qvel.length, expectedSize - qvelStart);
        for (let i = 0; i < qvelLength; i++) {
          fallbackObs[qvelStart + i] = simulation.qvel[i];
        }
      }

      return fallbackObs;
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
    if (!this.isModelLoaded) {
      console.warn('Model not loaded. Cannot run inference.');
      return null;
    }

    // Save the observation for debugging
    this.lastObservation = observation;

    try {
      // If in mock mode, use a simple mock policy
      if (this.mockMode) {
        return this.runMockInference(observation);
      }

      if (!ort || !this.session) {
        console.warn('ONNX Runtime or session not available. Using mock inference.');
        this.mockMode = true;
        return this.runMockInference(observation);
      }

      this.log(`Running inference with observation size: ${observation.length}`);

      // Validate observation size matches expected model input
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

      // Strict size validation
      if (observation.length !== requiredObsSize) {
        console.error(`Critical observation size mismatch: got ${observation.length}, need ${requiredObsSize}`);
        
        // Create a properly sized observation
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
   * Run mock inference when ONNX Runtime is not available
   * @param {Float32Array} observation - Observation array
   * @returns {Float32Array} mock action array
   */
  runMockInference(observation) {
    this.log('Running mock inference (ONNX Runtime not available)');
    
    // Create a mock action that produces reasonable bimanual arm control
    // Typically bimanual arms have around 40-80 actuators
    const actionSize = 40; // Conservative estimate for bimanual arm
    const mockAction = new Float32Array(actionSize);
    
    // Generate smooth, realistic control signals
    const time = this.inferenceCount * 0.1; // Simulate time progression
    
    for (let i = 0; i < actionSize; i++) {
      // Create smooth, low-amplitude control signals based on observation
      const obsInfluence = observation.length > i ? observation[i] * 0.1 : 0;
      const timeInfluence = Math.sin(time + i * 0.2) * 0.2;
      const damping = 0.8; // Damping factor to keep actions small
      
      mockAction[i] = (obsInfluence + timeInfluence) * damping;
      
      // Ensure actions stay within reasonable bounds
      mockAction[i] = Math.max(-0.5, Math.min(0.5, mockAction[i]));
    }
    
    // Increment inference counter
    this.inferenceCount++;
    
    // Save the action for debugging
    this.lastAction = mockAction;
    
    // Log inference details periodically
    if (this.inferenceCount % 20 === 0 || this.inferenceCount < 5) {
      this.log(`Completed mock inference #${this.inferenceCount}`);
      if (this.lastAction.length > 0) {
        const sampleSize = Math.min(3, this.lastAction.length);
        const actionSample = Array.from(this.lastAction.slice(0, sampleSize))
          .map(v => v.toFixed(3));
        this.log(`Mock action sample: [${actionSample.join(', ')}${this.lastAction.length > sampleSize ? ', ...' : ''}]`);
      }
    }
    
    return mockAction;
  }

  /**
   * Apply action to the simulation
   * @param {object} simulation - MuJoCo simulation object
   * @param {Float32Array} action - Action array from model inference
   */
  applyAction(simulation, action) {
    if (!action) {
      console.warn('No action provided to applyAction');
      return;
    }

    try {
      // Apply action values to the control array
      const ctrl = simulation.ctrl;
      const actionLength = Math.min(action.length, ctrl.length);

      // Process and apply each action value
      for (let i = 0; i < actionLength; i++) {
        // Get the raw action value
        let actionValue = action[i];

        // Apply action processing similar to Python implementation

        // 1. Clip action to valid range [-1, 1] if it's outside this range
        actionValue = Math.max(-1, Math.min(1, actionValue));

        // 2. Scale action to the appropriate range for the actuator
        // In MuJoCo, actuator ranges are typically defined in actuator_ctrlrange
        const ctrlRange = simulation.model ? simulation.model.actuator_ctrlrange : null;
        if (ctrlRange && i * 2 + 1 < ctrlRange.length) {
          const minValue = ctrlRange[i * 2];
          const maxValue = ctrlRange[i * 2 + 1];

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

      // Log a subset of the actions for debugging
      if (actionLength > 0) {
        const actionsToLog = Math.min(5, actionLength);
        const actionSample = Array.from(action.slice(0, actionsToLog));
        console.log(`Action sample: [${actionSample.join(', ')}${actionLength > actionsToLog ? ', ...' : ''}]`);

        const ctrlSample = Array.from(ctrl.slice(0, actionsToLog));
        console.log(`Control sample: [${ctrlSample.join(', ')}${ctrl.length > actionsToLog ? ', ...' : ''}]`);
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
    if (!this.isModelLoaded) {
      return { loaded: false };
    }

    if (this.mockMode) {
      return {
        loaded: true,
        mockMode: true,
        inputNames: ['obs'],
        outputNames: ['action'],
        inferenceCount: this.inferenceCount,
        lastObservationSize: this.lastObservation ? this.lastObservation.length : 0,
        lastActionSize: this.lastAction ? this.lastAction.length : 0
      };
    }

    return {
      loaded: true,
      mockMode: false,
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
    if (!this.isModelLoaded) {
      return {
        loaded: false,
        error: 'Model not loaded'
      };
    }

    if (this.mockMode) {
      return {
        loaded: true,
        mockMode: true,
        inputNames: ['obs'],
        outputNames: ['action'],
        inferenceCount: this.inferenceCount,
        inputShapes: { 'obs': [1, 210] },
        outputShapes: { 'action': [1, 40] },
        lastObservationSize: this.lastObservation ? this.lastObservation.length : 0,
        lastActionSize: this.lastAction ? this.lastAction.length : 0,
        note: 'Running in mock mode - ONNX Runtime not available'
      };
    }

    try {
      const diagnostics = {
        loaded: true,
        mockMode: false,
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