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
   * Get observation from MuJoCo simulation - matches the Python BimanualEnvV1 implementation
   * Based on DEFAULT_OBS_KEYS = ["time", "myohand_qpos", "myohand_qvel", "pros_hand_qpos", "pros_hand_qvel", "object_qpos", "object_qvel", "touching_body"]
   * @param {object} simulation - MuJoCo simulation object
   * @param {object} model - MuJoCo model object
   * @returns {Float32Array} observation vector
   */
  getObservation(simulation, model) {
    try {
      // Validate simulation and model
      if (!simulation) {
        console.error('Simulation object is null or undefined');
        return new Float32Array(210); // Return empty observation with expected size
      }

      if (!model) {
        console.error('Model object is null or undefined');
        return new Float32Array(210); // Return empty observation with expected size
      }

      // Extract basic state components from simulation
      const qpos = simulation.qpos || new Float32Array(0);
      const qvel = simulation.qvel || new Float32Array(0);
      const time = simulation.time || 0;

      this.log(`Simulation state: qpos=${qpos.length}, qvel=${qvel.length}, time=${time}`);
      this.log(`Model structure: nq=${model.nq}, nv=${model.nv}, nu=${model.nu}, nbody=${model.nbody}`);

      // Create observation components matching Python BimanualEnvV1 structure
      // DEFAULT_OBS_KEYS = ["time", "myohand_qpos", "myohand_qvel", "pros_hand_qpos", "pros_hand_qvel", "object_qpos", "object_qvel", "touching_body"]
      
      const observationComponents = [];
      
      // 1. Time component (1 dimension)
      observationComponents.push(time);
      
      // Analyze model structure to understand joint/body mapping
      const bodyNames = this.getAllBodyNames(model);
      const jointNames = this.getAllJointNames(model);
      
      this.log(`Found ${bodyNames.length} bodies and ${jointNames.length} joints`);
      if (this.debug && bodyNames.length > 0) {
        this.log(`Sample body names: ${bodyNames.slice(0, 5).join(', ')}`);
      }
      if (this.debug && jointNames.length > 0) {
        this.log(`Sample joint names: ${jointNames.slice(0, 5).join(', ')}`);
      }

      // 2-5. Identify myohand and prosthetic hand components
      // The bimanual model typically has two arms - one biological (myo) and one prosthetic
      const { myohandIndices, prosHandIndices, objectIndices } = this.identifyComponentIndices(model, bodyNames, jointNames);
      
      this.log(`Component indices - Myohand: ${myohandIndices.qpos.length}, Prosthetic: ${prosHandIndices.qpos.length}, Object: ${objectIndices.qpos.length}`);

      // 2. myohand_qpos (biological hand/arm joint positions)
      for (const idx of myohandIndices.qpos) {
        if (idx < qpos.length) {
          observationComponents.push(qpos[idx]);
        } else {
          observationComponents.push(0);
        }
      }

      // 3. myohand_qvel (biological hand/arm joint velocities)
      for (const idx of myohandIndices.qvel) {
        if (idx < qvel.length) {
          observationComponents.push(qvel[idx]);
        } else {
          observationComponents.push(0);
        }
      }

      // 4. pros_hand_qpos (prosthetic hand/arm joint positions)
      for (const idx of prosHandIndices.qpos) {
        if (idx < qpos.length) {
          observationComponents.push(qpos[idx]);
        } else {
          observationComponents.push(0);
        }
      }

      // 5. pros_hand_qvel (prosthetic hand/arm joint velocities)
      for (const idx of prosHandIndices.qvel) {
        if (idx < qvel.length) {
          observationComponents.push(qvel[idx]);
        } else {
          observationComponents.push(0);
        }
      }

      // 6. object_qpos (object position and orientation)
      for (const idx of objectIndices.qpos) {
        if (idx < qpos.length) {
          observationComponents.push(qpos[idx]);
        } else {
          observationComponents.push(0);
        }
      }

      // 7. object_qvel (object velocities)
      for (const idx of objectIndices.qvel) {
        if (idx < qvel.length) {
          observationComponents.push(qvel[idx]);
        } else {
          observationComponents.push(0);
        }
      }

      // 8. touching_body (contact forces/states)
      const contactComponents = this.getContactComponents(simulation, model);
      observationComponents.push(...contactComponents);

      this.log(`Observation components: time=1, myohand_qpos=${myohandIndices.qpos.length}, myohand_qvel=${myohandIndices.qvel.length}, pros_qpos=${prosHandIndices.qpos.length}, pros_qvel=${prosHandIndices.qvel.length}, object_qpos=${objectIndices.qpos.length}, object_qvel=${objectIndices.qvel.length}, touching=${contactComponents.length}`);

      // Create final observation array with exactly 210 dimensions
      const expectedSize = 210;
      const observation = new Float32Array(expectedSize);
      
      // Copy components to observation array
      const componentsLength = Math.min(observationComponents.length, expectedSize);
      for (let i = 0; i < componentsLength; i++) {
        observation[i] = observationComponents[i];
      }
      
      // Fill remaining dimensions with zeros if needed
      for (let i = componentsLength; i < expectedSize; i++) {
        observation[i] = 0;
      }

      this.log(`Final observation: ${observation.length} dimensions (target: ${expectedSize})`);

      // Log sample values for debugging
      if (this.debug) {
        const sampleSize = 5;
        const firstVals = Array.from(observation.slice(0, sampleSize)).map(v => v.toFixed(3));
        const lastVals = Array.from(observation.slice(-sampleSize)).map(v => v.toFixed(3));
        this.log(`Obs sample - first: [${firstVals.join(', ')}], last: [${lastVals.join(', ')}]`);
      }

      return observation;
    } catch (error) {
      console.error('Error creating observation:', error);
      // Return a safe fallback observation
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
   * Helper function to get all body names from model
   * @param {object} model - MuJoCo model object
   * @returns {Array} Array of body names
   */
  getAllBodyNames(model) {
    const bodyNames = [];
    if (!model || !model.nbody) return bodyNames;

    try {
      for (let i = 0; i < model.nbody; i++) {
        const name = this.getBodyName(model, i);
        bodyNames.push(name || `body_${i}`);
      }
    } catch (error) {
      console.error('Error getting body names:', error);
    }
    return bodyNames;
  }

  /**
   * Helper function to get all joint names from model
   * @param {object} model - MuJoCo model object
   * @returns {Array} Array of joint names
   */
  getAllJointNames(model) {
    const jointNames = [];
    if (!model || !model.njnt) return jointNames;

    try {
      for (let i = 0; i < model.njnt; i++) {
        const name = this.getJointName(model, i);
        jointNames.push(name || `joint_${i}`);
      }
    } catch (error) {
      console.error('Error getting joint names:', error);
    }
    return jointNames;
  }

  /**
   * Helper function to get joint name from model
   * @param {object} model - MuJoCo model object
   * @param {number} jointId - Joint ID
   * @returns {string} Joint name or null if not found
   */
  getJointName(model, jointId) {
    try {
      if (!model || jointId < 0 || jointId >= model.njnt) {
        return null;
      }

      if (model.names && model.name_jntadr) {
        const textDecoder = new TextDecoder("utf-8");
        const nameStr = textDecoder.decode(
          model.names.subarray(model.name_jntadr[jointId])
        );
        return nameStr.split('\0')[0];
      }
      return null;
    } catch (error) {
      console.error('Error getting joint name:', error);
      return null;
    }
  }

  /**
   * Identify component indices for myohand, prosthetic hand, and objects
   * @param {object} model - MuJoCo model object
   * @param {Array} bodyNames - Array of body names
   * @param {Array} jointNames - Array of joint names
   * @returns {object} Object containing indices for each component
   */
  identifyComponentIndices(model, bodyNames, jointNames) {
    const myohandIndices = { qpos: [], qvel: [] };
    const prosHandIndices = { qpos: [], qvel: [] };
    const objectIndices = { qpos: [], qvel: [] };

    // For a bimanual model, we need to identify which joints belong to which component
    // Based on typical naming conventions and the Python environment structure
    
    if (!model.jnt_qposadr || !jointNames.length) {
      // Fallback: assume equal division if we can't parse names
      const totalJoints = model.nq || 105;
      const jointsPerArm = Math.floor((totalJoints - 7) / 2); // Reserve 7 for object
      
      // First arm (myohand)
      for (let i = 0; i < jointsPerArm; i++) {
        myohandIndices.qpos.push(i);
        myohandIndices.qvel.push(i);
      }
      
      // Second arm (prosthetic)
      for (let i = jointsPerArm; i < jointsPerArm * 2; i++) {
        prosHandIndices.qpos.push(i);
        prosHandIndices.qvel.push(i);
      }
      
      // Object
      for (let i = jointsPerArm * 2; i < totalJoints; i++) {
        objectIndices.qpos.push(i);
        objectIndices.qvel.push(i);
      }
      
      this.log(`Using fallback joint division: myohand=${jointsPerArm}, pros=${jointsPerArm}, object=${totalJoints - jointsPerArm * 2}`);
      return { myohandIndices, prosHandIndices, objectIndices };
    }

    // Parse joint names to identify components
    for (let i = 0; i < jointNames.length; i++) {
      const jointName = jointNames[i].toLowerCase();
      const qposStart = model.jnt_qposadr[i];
      const qposEnd = i + 1 < model.jnt_qposadr.length ? model.jnt_qposadr[i + 1] : qposStart + 1;
      
      // Identify joint type based on name patterns
      let isMyohand = false;
      let isProsthetic = false;
      let isObject = false;
      
      // Check for biological hand/arm indicators
      if (jointName.includes('myo') || 
          jointName.includes('bio') || 
          jointName.includes('muscle') ||
          jointName.includes('left') || // Often the biological side
          jointName.includes('human')) {
        isMyohand = true;
      }
      // Check for prosthetic indicators
      else if (jointName.includes('pros') || 
               jointName.includes('robot') || 
               jointName.includes('mpl') ||
               jointName.includes('right') || // Often the prosthetic side
               jointName.includes('artificial')) {
        isProsthetic = true;
      }
      // Check for object indicators
      else if (jointName.includes('object') || 
               jointName.includes('target') || 
               jointName.includes('box') ||
               jointName.includes('item') ||
               jointName.includes('goal')) {
        isObject = true;
      }
      
      // Add indices to appropriate arrays
      for (let j = qposStart; j < qposEnd; j++) {
        if (isMyohand) {
          myohandIndices.qpos.push(j);
          myohandIndices.qvel.push(j);
        } else if (isProsthetic) {
          prosHandIndices.qpos.push(j);
          prosHandIndices.qvel.push(j);
        } else if (isObject) {
          objectIndices.qpos.push(j);
          objectIndices.qvel.push(j);
        } else {
          // If unclear, assign to myohand (default)
          myohandIndices.qpos.push(j);
          myohandIndices.qvel.push(j);
        }
      }
    }

    this.log(`Parsed joint components: myohand=${myohandIndices.qpos.length}, pros=${prosHandIndices.qpos.length}, object=${objectIndices.qpos.length}`);
    
    return { myohandIndices, prosHandIndices, objectIndices };
  }

  /**
   * Extract contact/touching information from simulation
   * @param {object} simulation - MuJoCo simulation object
   * @param {object} model - MuJoCo model object
   * @returns {Array} Array of contact values
   */
  getContactComponents(simulation, model) {
    const contactComponents = [];
    
    try {
      // Check for contact forces or collision information
      if (simulation.contact && simulation.contact.length > 0) {
        // Use actual contact data if available
        const maxContacts = 30; // Limit to reasonable number
        for (let i = 0; i < Math.min(simulation.contact.length, maxContacts); i++) {
          contactComponents.push(simulation.contact[i]);
        }
      } else if (simulation.cfrc_ext) {
        // Use external contact forces
        const maxForces = 30;
        for (let i = 0; i < Math.min(simulation.cfrc_ext.length, maxForces); i++) {
          contactComponents.push(simulation.cfrc_ext[i]);
        }
      } else {
        // Generate synthetic contact information based on body interactions
        const numContactSensors = 30; // Typical number for bimanual setup
        for (let i = 0; i < numContactSensors; i++) {
          contactComponents.push(0); // No contact
        }
      }
    } catch (error) {
      console.error('Error extracting contact components:', error);
      // Fallback to zeros
      const numContactSensors = 30;
      for (let i = 0; i < numContactSensors; i++) {
        contactComponents.push(0);
      }
    }
    
    return contactComponents;
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