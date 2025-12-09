import { useState, useEffect } from 'react';
import {
  Container,
  Box,
  CssBaseline,
  ThemeProvider,
  createTheme,
  Alert,
  Snackbar,
  Paper,
  Grid,
  Button,
  Typography,
  TextField,
  MenuItem,
  Tabs,
  Tab,
  FormControl,
  InputLabel,
  Select,
} from '@mui/material';
import {type SelectChangeEvent } from '@mui/material';
import { 
  School, 
  TrendingUp, 
  Assessment, 
  Warning as WarningIcon, 
} from '@mui/icons-material';
import apiService from './services/apiService.js';
import Header from './components/Header.jsx';
import ResultCard from './components/ResultCard.jsx';
import Footer from './components/Footer.jsx';

// Types
type TabType = 'persistence' | 'dropout' | 'gpa' | 'success';

interface FormData {
  First_Term_GPA: number;
  Second_Term_GPA?: number;
  First_Language: number;
  Funding: number;
  FastTrack: number;
  Coop: number;
  Residency: number;
  Gender: number;
  Prev_Education: number;
  Age_Group: number;
  HS_Average: number;
  Math_Score: number;
  English_Grade: number;
}

// Theme
const theme = createTheme({
  palette: {
    primary: { main: '#667eea' },
    secondary: { main: '#764ba2' }
  },
  typography: {
    fontFamily: '"Segoe UI", "Roboto", "Helvetica", "Arial", sans-serif'
  }
});

// Form Input Component
interface FormInputProps {
  label: string;
  name: keyof FormData;
  value: number;
  onChange: (name: keyof FormData, value: number) => void;
  type?: string;
  options?: { value: number; label: string }[];
  step?: number;
  min?: number;
  max?: number;
}

const FormInput: React.FC<FormInputProps> = ({ 
  label, 
  name, 
  value, 
  onChange, 
  type = 'number', 
  options = null, 
  step = 1, 
  min, 
  max 
}) => {
  if (options) {
    return (
      <FormControl fullWidth>
        <InputLabel>{label}</InputLabel>
        <Select
          value={value}
          label={label}
          onChange={(e: SelectChangeEvent<number>) => onChange(name, Number(e.target.value))}
        >
          {options.map((option) => (
            <MenuItem key={option.value} value={option.value}>
              {option.label}
            </MenuItem>
          ))}
        </Select>
      </FormControl>
    );
  }

  return (
    <TextField
      fullWidth
      label={label}
      type={type}
      value={value}
      onChange={(e) => onChange(name, Number(e.target.value))}
      inputProps={{ step, min, max }}
      variant="outlined"
    />
  );
};

function App() {
  const [activeTab, setActiveTab] = useState<TabType>('persistence');
  const [apiStatus, setApiStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking');
  const [loading, setLoading] = useState(false);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'info' as 'info' | 'success' | 'error' });
  const [predictions, setPredictions] = useState<Record<TabType, any>>({
    persistence: null,
    dropout: null,
    gpa: null,
    success: null
  });

  // Form States - Using NUMERIC CODES as per CSV
  const defaultForm: FormData = {
    First_Term_GPA: 3.0,
    Second_Term_GPA: 3.0,
    First_Language: 1, // 1 = English
    Funding: 2, // 2 = GPOG_FT
    FastTrack: 2, // 2 = N
    Coop: 2, // 2 = N
    Residency: 1, // 1 = Domestic
    Gender: 2, // 2 = Male
    Prev_Education: 1, // 1 = HighSchool
    Age_Group: 3, // 3 = 21 to 25
    HS_Average: 80.0,
    Math_Score: 25.0,
    English_Grade: 7 // 7 = Level-160
  };

  const [forms, setForms] = useState<Record<TabType, FormData>>({
    persistence: { ...defaultForm },
    dropout: { ...defaultForm },
    gpa: { ...defaultForm },
    success: { ...defaultForm }
  });

  useEffect(() => {
    checkApiHealth();
    const interval = setInterval(checkApiHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const checkApiHealth = async () => {
    const result = await apiService.checkHealth();
    setApiStatus(result.success ? 'connected' : 'disconnected');
  };

  const showSnackbar = (message: string, severity: 'info' | 'success' | 'error' = 'info') => {
    setSnackbar({ open: true, message, severity });
  };

  const handlePredict = async (endpoint: TabType) => {
    if (apiStatus !== 'connected') {
      showSnackbar('API is not connected. Please check backend server.', 'error');
      return;
    }

    setLoading(true);
    const result = await apiService.predict(endpoint, forms[endpoint]);
    setLoading(false);

    if (result.success) {
      setPredictions(prev => ({ ...prev, [endpoint]: result.data }));
      showSnackbar('Prediction completed successfully!', 'success');
    } else {
      showSnackbar(`Error: ${result.error}`, 'error');
    }
  };

  const updateForm = (tab: TabType, name: keyof FormData, value: number) => {
    setForms(prev => ({
      ...prev,
      [tab]: { ...prev[tab], [name]: value }
    }));
  };

  // Options based on CSV encoding
  const options = {
    language: [
      { value: 1, label: 'English' },
      { value: 2, label: 'French' },
      { value: 3, label: 'Other' }
    ],
    funding: [
      { value: 1, label: 'Apprentice_PS' },
      { value: 2, label: 'GPOG_FT' },
      { value: 3, label: 'Intl Offshore' },
      { value: 4, label: 'Intl Regular' },
      { value: 5, label: 'Intl Transfer' },
      { value: 6, label: 'Joint Program Ryerson' },
      { value: 7, label: 'Joint Program UTSC' },
      { value: 8, label: 'Second Career' },
      { value: 9, label: 'WSIB' }
    ],
    yesNo: [
      { value: 1, label: 'Yes' },
      { value: 2, label: 'No' }
    ],
    residency: [
      { value: 1, label: 'Domestic' },
      { value: 2, label: 'International' }
    ],
    gender: [
      { value: 1, label: 'Female' },
      { value: 2, label: 'Male' },
    ],
    education: [
      { value: 1, label: 'High School' },
      { value: 2, label: 'Post-Secondary' }
    ],
    age: [
      { value: 1, label: '0-18' },
      { value: 2, label: '19-20' },
      { value: 3, label: '21-25' },
      { value: 4, label: '26-30' },
      { value: 5, label: '31-35' },
      { value: 6, label: '36-40' },
      { value: 7, label: '41-50' },
      { value: 8, label: '51-60' },
      { value: 9, label: '61-65' },
      { value: 10, label: '66+' }
    ],
    englishGrade: [
      { value: 1, label: 'Level-130' },
      { value: 2, label: 'Level-131' },
      { value: 3, label: 'Level-140' },
      { value: 4, label: 'Level-141' },
      { value: 5, label: 'Level-150' },
      { value: 6, label: 'Level-151' },
      { value: 7, label: 'Level-160' },
      { value: 8, label: 'Level-161' },
      { value: 9, label: 'Level-170' },
      { value: 10, label: 'Level-171' },
      { value: 11, label: 'Level-180' }
    ]
  };

  const renderCommonFields = (tab: TabType) => (
    <>
      <Grid item xs={12} sm={6} md={4}>
        <FormInput label="First Term GPA (0-4.5)" name="First_Term_GPA" value={forms[tab].First_Term_GPA} onChange={(n, v) => updateForm(tab, n, v)} step={0.1} min={0} max={4.5} />
      </Grid>
      <Grid item xs={12} sm={6} md={4}>
        <FormInput label="HS Average (0-100)" name="HS_Average" value={forms[tab].HS_Average} onChange={(n, v) => updateForm(tab, n, v)} min={0} max={100} />
      </Grid>
      <Grid item xs={12} sm={6} md={4}>
        <FormInput label="Math Score (0-50)" name="Math_Score" value={forms[tab].Math_Score} onChange={(n, v) => updateForm(tab, n, v)} min={0} max={50} />
      </Grid>
      <Grid item xs={12} sm={6} md={4}>
        <FormInput label="First Language" name="First_Language" value={forms[tab].First_Language} onChange={(n, v) => updateForm(tab, n, v)} options={options.language} />
      </Grid>
      <Grid item xs={12} sm={6} md={4}>
        <FormInput label="Funding" name="Funding" value={forms[tab].Funding} onChange={(n, v) => updateForm(tab, n, v)} options={options.funding} />
      </Grid>
      <Grid item xs={12} sm={6} md={4}>
        <FormInput label="Fast Track" name="FastTrack" value={forms[tab].FastTrack} onChange={(n, v) => updateForm(tab, n, v)} options={options.yesNo} />
      </Grid>
      <Grid item xs={12} sm={6} md={4}>
        <FormInput label="Co-op" name="Coop" value={forms[tab].Coop} onChange={(n, v) => updateForm(tab, n, v)} options={options.yesNo} />
      </Grid>
      <Grid item xs={12} sm={6} md={4}>
        <FormInput label="Residency" name="Residency" value={forms[tab].Residency} onChange={(n, v) => updateForm(tab, n, v)} options={options.residency} />
      </Grid>
      <Grid item xs={12} sm={6} md={4}>
        <FormInput label="Gender" name="Gender" value={forms[tab].Gender} onChange={(n, v) => updateForm(tab, n, v)} options={options.gender} />
      </Grid>
      <Grid item xs={12} sm={6} md={4}>
        <FormInput label="Previous Education" name="Prev_Education" value={forms[tab].Prev_Education} onChange={(n, v) => updateForm(tab, n, v)} options={options.education} />
      </Grid>
      <Grid item xs={12} sm={6} md={4}>
        <FormInput label="Age Group" name="Age_Group" value={forms[tab].Age_Group} onChange={(n, v) => updateForm(tab, n, v)} options={options.age} />
      </Grid>
      <Grid item xs={12} sm={6} md={4}>
        <FormInput label="English Grade" name="English_Grade" value={forms[tab].English_Grade} onChange={(n, v) => updateForm(tab, n, v)} options={options.englishGrade} />
      </Grid>
    </>
  );

  const tabs = [
    { id: 'persistence' as TabType, label: 'Persistence', icon: <TrendingUp />, model: 'Model 1: Hybrid RF+NN' },
    { id: 'dropout' as TabType, label: 'Dropout Risk', icon: <WarningIcon />, model: 'Model 2: NN Classifier' },
    { id: 'gpa' as TabType, label: 'GPA Prediction', icon: <Assessment />, model: 'Model 3: Regression NN' },
    { id: 'success' as TabType, label: 'Program Success', icon: <School />, model: 'Model 4: NN Classifier' }
  ];

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
        <Header apiStatus={apiStatus} />

        <Box sx={{ borderBottom: 1, borderColor: 'divider', bgcolor: 'white' }}>
          <Tabs value={activeTab} onChange={(_, v) => setActiveTab(v)} variant="fullWidth" sx={{ '& .MuiTab-root': { fontSize: '1rem', fontWeight: 600, textTransform: 'none', py: 2 } }}>
            {tabs.map(tab => (
              <Tab key={tab.id} icon={tab.icon} iconPosition="start" label={tab.label} value={tab.id} />
            ))}
          </Tabs>
        </Box>

        <Container maxWidth="lg" sx={{ flex: 1, py: 4 }}>
          <Paper sx={{ p: 4 }}>
            <Box sx={{ mb: 3 }}>
              <Typography variant="h4" fontWeight="bold" gutterBottom>
                {tabs.find(t => t.id === activeTab)?.label}
              </Typography>
              <Typography variant="body1" color="text.secondary">
                {tabs.find(t => t.id === activeTab)?.model}
              </Typography>
            </Box>

            <Grid container spacing={3}>
              {activeTab === 'persistence' && (
                <Grid item xs={12} sm={6} md={4}>
                  <FormInput label="Second Term GPA (0-4.5)" name="Second_Term_GPA" value={forms.persistence.Second_Term_GPA!} onChange={(n, v) => updateForm('persistence', n, v)} step={0.1} min={0} max={4.5} />
                </Grid>
              )}
              {renderCommonFields(activeTab)}
            </Grid>

            <Button
              fullWidth
              variant="contained"
              size="large"
              startIcon={tabs.find(t => t.id === activeTab)?.icon}
              onClick={() => handlePredict(activeTab)}
              disabled={loading}
              sx={{ 
                mt: 4, 
                py: 1.5, 
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 
                fontSize: '1.1rem', 
                fontWeight: 'bold' 
              }}
            >
              {loading ? 'Predicting...' : `Predict ${tabs.find(t => t.id === activeTab)?.label}`}
            </Button>

            {predictions[activeTab] && (
              <ResultCard 
                title={`${tabs.find(t => t.id === activeTab)?.label} Result`} 
                data={predictions[activeTab]} 
                type={activeTab} 
              />
            )}
          </Paper>
        </Container>

        <Footer />

        <Snackbar 
          open={snackbar.open} 
          autoHideDuration={4000} 
          onClose={() => setSnackbar({...snackbar, open: false})} 
          anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        >
          <Alert severity={snackbar.severity} sx={{ width: '100%' }} onClose={() => setSnackbar({...snackbar, open: false})}>
            {snackbar.message}
          </Alert>
        </Snackbar>
      </Box>
    </ThemeProvider>
  );
}

export default App;