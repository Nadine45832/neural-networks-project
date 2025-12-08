// Minimal module declarations for existing .jsx components and JS services
// This prevents TypeScript from treating imports from these files as `any`.

declare module './components/PredictionTabs' {
  import React from 'react';
  const PredictionTabs: React.ComponentType<any>;
  export default PredictionTabs;
}

declare module './components/SuccessForm' {
  import React from 'react';
  const SuccessForm: React.ComponentType<any>;
  export default SuccessForm;
}

declare module './components/PersistenceForm' {
  import React from 'react';
  const PersistenceForm: React.ComponentType<any>;
  export default PersistenceForm;
}

declare module './components/GpaForm' {
  import React from 'react';
  const GpaForm: React.ComponentType<any>;
  export default GpaForm;
}

declare module './components/Footer' {
  import React from 'react';
  const Footer: React.ComponentType<any>;
  export default Footer;
}

declare module './components/Header' {
  import React from 'react';
  const Header: React.ComponentType<any>;
  export default Header;
}

declare module './components/FormInput' {
  import React from 'react';
  const FormInput: React.ComponentType<any>;
  export default FormInput;
}

declare module './components/ResultCard' {
  import React from 'react';
  const ResultCard: React.ComponentType<any>;
  export default ResultCard;
}

declare module './services/apiService' {
  export interface HealthResult { success: boolean }
  export interface PredictResult { success: boolean; data?: any; error?: string }
  const apiService: {
    checkHealth(): Promise<HealthResult>;
    predict(endpoint: string, data: any): Promise<PredictResult>;
  };
  export default apiService;
}

// Also cover imports that include the .js extension or different relative paths
declare module './services/apiService.js' {
  import { HealthResult, PredictResult } from './services/apiService';
  const apiService: {
    checkHealth(): Promise<HealthResult>;
    predict(endpoint: string, data: any): Promise<PredictResult>;
  };
  export default apiService;
}

declare module 'services/apiService' {
  import { HealthResult, PredictResult } from './services/apiService';
  const apiService: {
    checkHealth(): Promise<HealthResult>;
    predict(endpoint: string, data: any): Promise<PredictResult>;
  };
  export default apiService;
}

// Fallback for any other .jsx imports (keeps strict typing but prevents errors)
declare module '*.jsx' {
  import React from 'react';
  const Component: React.ComponentType<any>;
  export default Component;
}

declare module '*.js' {
  import React from 'react';
  const value: any;
  export default value;
}
