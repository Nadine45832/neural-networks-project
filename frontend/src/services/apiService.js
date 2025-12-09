const API_BASE_URL = 'http://localhost:5001/api';

const apiService = {
  async checkHealth() {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      const data = await response.json();
      return { success: true, data };
    } catch (error) {
      return { success: false, error: error.message };
    }
  },

  async predict(endpoint, formData) {
    try {
      const response = await fetch(`${API_BASE_URL}/predict/${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Prediction failed');
      }
      
      return { success: true, data };
    } catch (error) {
      return { success: false, error: error.message };
    }
  },

  async getModelsInfo() {
    try {
      const response = await fetch(`${API_BASE_URL}/models/info`);
      const data = await response.json();
      return { success: true, data };
    } catch (error) {
      return { success: false, error: error.message };
    }
  },

  async predictComprehensive(formData) {
    try {
      const response = await fetch(`${API_BASE_URL}/predict/comprehensive`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Comprehensive prediction failed');
      }
      
      return { success: true, data };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }
};

export default apiService;