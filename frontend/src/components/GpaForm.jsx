import React from 'react';
import { Box, Grid, Button, Typography, Paper } from '@mui/material';
import { Assessment } from '@mui/icons-material';
import FormInput from './FormInput';
import ResultCard from './ResultCard';

const GpaForm = ({ formData, onChange, onSubmit, loading, result }) => {
  return (
    <Paper sx={{ p: 4 }}>
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" fontWeight="bold" gutterBottom>
          GPA Prediction
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Predict second term GPA based on performance indicators
        </Typography>
      </Box>

      <Grid container spacing={3}>
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
            label="First Term GPA (0-4)"
            name="first_term_gpa"
            value={formData.first_term_gpa}
            onChange={onChange}
            step={0.1}
            min={0}
            max={4}
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
            label="Assignment Completion (%)"
            name="assignment_completion"
            value={formData.assignment_completion}
            onChange={onChange}
            min={0}
            max={100}
          />
        </Grid>

        <Grid item xs={12} sm={6} md={4}>
          <FormInput
            label="Participation Score (1-10)"
            name="participation_score"
            value={formData.participation_score}
            onChange={onChange}
            min={1}
            max={10}
          />
        </Grid>
      </Grid>

      <Button
        fullWidth
        variant="contained"
        size="large"
        startIcon={<Assessment />}
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
        {loading ? 'Predicting...' : 'Predict GPA'}
      </Button>

      {result && (
        <ResultCard 
          title="Prediction Result"
          data={result}
          type="gpa"
        />
      )}
    </Paper>
  );
};

export default GpaForm;