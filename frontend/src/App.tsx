import { useState, useEffect } from 'react';
import {
  Container,
  Box,
  CssBaseline,
  ThemeProvider,
  createTheme,
  Alert,
  Snackbar
} from '@mui/material';
import PredictionTabs from './components/PersistenceForm.jsx';
import SuccessForm from './components/SuccessForm.jsx';
import PersistenceForm from './components/PersistenceForm.jsx';
import GpaForm from './components/GpaForm.jsx';
import Footer from './components/Footer.jsx';
import apiService from './services/apiService.js';
import Header from "./components/Header.jsx";

// ---------- Types ----------
type TabType = 'success' | 'persistence' | 'gpa';

interface SnackbarState {
  open: boolean;
  message: string;
  severity: 'info' | 'success' | 'warning' | 'error';
}

interface SuccessFormData {
  age: number;
  high_school_gpa: number;
  attendance_rate: number;
  study_hours: number;
  family_support: number;
  extracurricular: number;
  financial_aid: number;
  work_hours: number;
}

interface PersistenceFormData {
  first_term_gpa: number;
  attendance_first_term: number;
  engagement_score: number;
  financial_difficulty: number;
  commute_time: number;
  social_integration: number;
  academic_support: number;
}

interface GpaFormData {
  high_school_gpa: number;
  first_term_gpa: number;
  study_hours: number;
  attendance_rate: number;
  assignment_completion: number;
  participation_score: number;
}

type FormData = SuccessFormData | PersistenceFormData | GpaFormData;

interface PredictionResults {
  success: number | null;
  persistence: number | null;
  gpa: number | null;
}

// ---------- THEME ----------
const theme = createTheme({
  palette: {
    primary: { main: '#667eea' },
    secondary: { main: '#764ba2' }
  },
  typography: {
    fontFamily: '"Segoe UI", "Roboto", "Helvetica", "Arial", sans-serif'
  }
});

// ------------------- COMPONENT -------------------
function App() {
  const [activeTab, setActiveTab] = useState<TabType>('success');
  const [apiStatus, setApiStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking');
  const [loading, setLoading] = useState(false);
  const [snackbar, setSnackbar] = useState<SnackbarState>({
    open: false,
    message: '',
    severity: 'info'
  });

  const [predictions, setPredictions] = useState<PredictionResults>({
    success: null,
    persistence: null,
    gpa: null
  });

  // ----- Forms -----
  const [successForm, setSuccessForm] = useState<SuccessFormData>({
    age: 20,
    high_school_gpa: 3.0,
    attendance_rate: 85,
    study_hours: 15,
    family_support: 1,
    extracurricular: 0,
    financial_aid: 1,
    work_hours: 10
  });

  const [persistenceForm, setPersistenceForm] = useState<PersistenceFormData>({
    first_term_gpa: 3.0,
    attendance_first_term: 90,
    engagement_score: 7,
    financial_difficulty: 0,
    commute_time: 30,
    social_integration: 5,
    academic_support: 1
  });

  const [gpaForm, setGpaForm] = useState<GpaFormData>({
    high_school_gpa: 3.0,
    first_term_gpa: 3.0,
    study_hours: 15,
    attendance_rate: 85,
    assignment_completion: 90,
    participation_score: 7
  });

  // ---------------- HEALTH CHECK ----------------
  useEffect(() => {
    checkApiHealth();
    const interval = setInterval(checkApiHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const checkApiHealth = async () => {
    const result = await apiService.checkHealth();
    setApiStatus(result.success ? 'connected' : 'disconnected');
  };

  const handleTabChange = (_: any, newValue: TabType) => {
    setActiveTab(newValue);
  };

  const showSnackbar = (message: string, severity: SnackbarState['severity'] = 'info') => {
    setSnackbar({ open: true, message, severity });
  };

  const handleCloseSnackbar = () => {
    setSnackbar(prev => ({ ...prev, open: false }));
  };

  // -------- UPDATE FORM ----------
  const handleFormChange = <T extends FormData>(
    formName: TabType,
    fieldName: keyof T,
    value: any
  ) => {
    const setters = {
      success: setSuccessForm,
      persistence: setPersistenceForm,
      gpa: setGpaForm
    };

    const forms = {
      success: successForm,
      persistence: persistenceForm,
      gpa: gpaForm
    };

    (setters[formName] as React.Dispatch<React.SetStateAction<T>>)({
      ...(forms[formName] as T),
      [fieldName]: value
    });
  };

  // -------- PREDICT ----------
  const handlePredict = async (endpoint: TabType, formData: FormData) => {
    if (apiStatus !== 'connected') {
      showSnackbar('API is not connected. Please check backend server.', 'error');
      return;
    }

    setLoading(true);
    const result = await apiService.predict(endpoint, formData);
    setLoading(false);

    if (result.success) {
      setPredictions(prev => ({ ...prev, [endpoint]: result.data }));
      showSnackbar('Prediction completed successfully!', 'success');
    } else {
      showSnackbar(`Error: ${result.error}`, 'error');
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          minHeight: '100vh',
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        }}
      >
        <Header apiStatus={apiStatus} />

        <PredictionTabs activeTab={activeTab} onChange={handleTabChange} />

        <Container maxWidth="lg" sx={{ flex: 1, py: 4 }}>
          {activeTab === 'success' && (
            <SuccessForm
              formData={successForm}
              onChange={(name: keyof SuccessFormData, value: any) =>
                handleFormChange<SuccessFormData>('success', name, value)
              }
              onSubmit={() => handlePredict('success', successForm)}
              loading={loading}
              result={predictions.success}
            />
          )}

          {activeTab === 'persistence' && (
            <PersistenceForm
              formData={persistenceForm}
              onChange={(name: keyof PersistenceFormData, value: any) =>
                handleFormChange<PersistenceFormData>('persistence', name, value)
              }
              onSubmit={() => handlePredict('persistence', persistenceForm)}
              loading={loading}
              result={predictions.persistence}
            />
          )}

          {activeTab === 'gpa' && (
            <GpaForm
              formData={gpaForm}
              onChange={(name: keyof GpaFormData, value: any) =>
                handleFormChange<GpaFormData>('gpa', name, value)
              }
              onSubmit={() => handlePredict('gpa', gpaForm)}
              loading={loading}
              result={predictions.gpa}
            />
          )}
        </Container>

        <Footer />

        <Snackbar
          open={snackbar.open}
          autoHideDuration={4000}
          onClose={handleCloseSnackbar}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        >
          <Alert severity={snackbar.severity} sx={{ width: '100%' }} onClose={handleCloseSnackbar}>
            {snackbar.message}
          </Alert>
        </Snackbar>
      </Box>
    </ThemeProvider>
  );
}

export default App;
