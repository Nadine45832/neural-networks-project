import React from 'react';
import { Box, Grid, Button, Typography, Paper } from '@mui/material';
import { TrendingUp } from '@mui/icons-material';
import FormInput from './FormInput';
import ResultCard from './ResultCard';

const PersistenceForm = ({ formData, onChange, onSubmit, loading, result }) => {
  if (!formData) {
    // Defensive fallback to avoid runtime crashes if prop is missing
    console.warn('PersistenceForm: received undefined formData, using defaults');
  }
  const fd = formData || {
    first_term_gpa: 0,
    attendance_first_term: 0,
    engagement_score: 1,
    financial_difficulty: 0,
    commute_time: 0,
    social_integration: 1,
    academic_support: 0
  };
  const yesNoOptions = [
    { value: 0, label: 'No' },
    { value: 1, label: 'Yes' }
  ];

  return (
    <Paper sx={{ p: 4 }}>
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" fontWeight="bold" gutterBottom>
          First Year Persistence Prediction
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Predict likelihood of student continuing to second year
        </Typography>
      </Box>

      <Grid container spacing={3}>
        <Grid item xs={12} sm={6} md={4}>
          <FormInput
            label="First Term GPA (0-4)"
            name="first_term_gpa"
            value={fd.first_term_gpa}
            onChange={onChange}
            step={0.1}
            min={0}
            max={4}
          />
        </Grid>

        <Grid item xs={12} sm={6} md={4}>
          <FormInput
            label="Attendance First Term (%)"
            name="attendance_first_term"
            value={fd.attendance_first_term}
            onChange={onChange}
            min={0}
            max={100}
          />
        </Grid>

        <Grid item xs={12} sm={6} md={4}>
          <FormInput
            label="Engagement Score (1-10)"
            name="engagement_score"
            value={fd.engagement_score}
            onChange={onChange}
            min={1}
            max={10}
          />
        </Grid>

        <Grid item xs={12} sm={6} md={4}>
          <FormInput
            label="Financial Difficulty"
            name="financial_difficulty"
            value={fd.financial_difficulty}
            onChange={onChange}
            options={yesNoOptions}
          />
        </Grid>

        <Grid item xs={12} sm={6} md={4}>
          <FormInput
            label="Commute Time (minutes)"
            name="commute_time"
            value={fd.commute_time}
            onChange={onChange}
            min={0}
            max={240}
          />
        </Grid>

        <Grid item xs={12} sm={6} md={4}>
          <FormInput
            label="Social Integration (1-10)"
            name="social_integration"
            value={fd.social_integration}
            onChange={onChange}
            min={1}
            max={10}
          />
        </Grid>

        <Grid item xs={12} sm={6} md={4}>
          <FormInput
            label="Academic Support"
            name="academic_support"
            value={fd.academic_support}
            onChange={onChange}
            options={yesNoOptions}
          />
        </Grid>
      </Grid>

      <Button
        fullWidth
        variant="contained"
        size="large"
        startIcon={<TrendingUp />}
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
        {loading ? 'Predicting...' : 'Predict Persistence'}
      </Button>

      {result && (
        <ResultCard 
          title="Prediction Result"
          data={result}
          type="persistence"
        />
      )}
    </Paper>
  );
};

export default PersistenceForm;