import React from 'react';
import { Box, Grid, Button, Typography, Paper } from '@mui/material';
import { Psychology } from '@mui/icons-material';
import FormInput from './FormInput';
import ResultCard from './ResultCard';

const SuccessForm = ({ formData, onChange, onSubmit, loading, result }) => {
  const yesNoOptions = [
    { value: 0, label: 'No' },
    { value: 1, label: 'Yes' }
  ];

  return (
    <Paper sx={{ p: 4 }}>
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" fontWeight="bold" gutterBottom>
          Student Success Prediction
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Predict program completion probability based on student characteristics
        </Typography>
      </Box>

      <Grid container spacing={3}>
        <Grid item xs={12} sm={6} md={4}>
          <FormInput
            label="Age"
            name="age"
            value={formData.age}
            onChange={onChange}
            min={18}
            max={65}
          />
        </Grid>

        <Grid item xs={12} sm={6} md={4}>
          <FormInput
            label="High School GPA (0-4)"
            name="high_school_gpa"
            value={formData.high_school_gpa}
            onChange={onChange}
            step={0.1}
            min={0}
            max={4}
          />
        </Grid>

        <Grid item xs={12} sm={6} md={4}>
          <FormInput
            label="Attendance Rate (%)"
            name="attendance_rate"
            value={formData.attendance_rate}
            onChange={onChange}
            min={0}
            max={100}
          />
        </Grid>

        <Grid item xs={12} sm={6} md={4}>
          <FormInput
            label="Study Hours/Week"
            name="study_hours"
            value={formData.study_hours}
            onChange={onChange}
            min={0}
            max={168}
          />
        </Grid>

        <Grid item xs={12} sm={6} md={4}>
          <FormInput
            label="Family Support"
            name="family_support"
            value={formData.family_support}
            onChange={onChange}
            options={yesNoOptions}
          />
        </Grid>

        <Grid item xs={12} sm={6} md={4}>
          <FormInput
            label="Extracurricular Activities"
            name="extracurricular"
            value={formData.extracurricular}
            onChange={onChange}
            options={yesNoOptions}
          />
        </Grid>

        <Grid item xs={12} sm={6} md={4}>
          <FormInput
            label="Financial Aid"
            name="financial_aid"
            value={formData.financial_aid}
            onChange={onChange}
            options={yesNoOptions}
          />
        </Grid>

        <Grid item xs={12} sm={6} md={4}>
          <FormInput
            label="Work Hours/Week"
            name="work_hours"
            value={formData.work_hours}
            onChange={onChange}
            min={0}
            max={40}
          />
        </Grid>
      </Grid>

      <Button
        fullWidth
        variant="contained"
        size="large"
        startIcon={<Psychology />}
        onClick={onSubmit}
        disabled={loading}
        sx={{ 
          mt: 4,
          py: 1.5,
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          fontSize: '1.1rem',
          fontWeight: 'bold',
          '&:hover': {
            background: 'linear-gradient(135deg, #5568d3 0%, #6a3f8f 100%)',
          }
        }}
      >
        {loading ? 'Predicting...' : 'Predict Success'}
      </Button>

      {result && (
        <ResultCard 
          title="Prediction Result"
          data={result}
          type="success"
        />
      )}
    </Paper>
  );
};

export default SuccessForm;