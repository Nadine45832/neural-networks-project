/**
 * API Service - Handles all API communication
 * Single Responsibility Principle: Only responsible for API calls
 */

// Use Vite dev proxy during development by using a relative `/api` path.
// In production builds you can set the full backend URL.
const API_BASE_URL = import.meta.env.DEV ? '/api' : 'http://localhost:5001/api';

class ApiService {
  /**
   * Check API health status
   */
  async checkHealth() {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      const data = await response.json();
      return { success: true, data };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  /**
   * Make a prediction request
   * @param {string} endpoint - Prediction endpoint (success, persistence, gpa)
   * @param {object} formData - Form data to send
   */
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
  }

  /**
   * Get model information
   */
  async getModelsInfo() {
    try {
      const response = await fetch(`${API_BASE_URL}/models/info`);
      const data = await response.json();
      return { success: true, data };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  /**
   * Batch prediction
   * @param {array} students - Array of student data
   */
  async predictBatch(students) {
    try {
      const response = await fetch(`${API_BASE_URL}/predict/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ students })
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Batch prediction failed');
      }
      
      return { success: true, data };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }
}

export default new ApiService();