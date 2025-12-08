import React from 'react';
import { Tabs, Tab, Box } from '@mui/material';
import { School, TrendingUp, Assessment } from '@mui/icons-material';

const PredictionTabs = ({ activeTab, onChange }) => {
  return (
    <Box sx={{ borderBottom: 1, borderColor: 'divider', bgcolor: 'white' }}>
      <Tabs 
        value={activeTab} 
        onChange={onChange}
        variant="fullWidth"
        sx={{
          '& .MuiTab-root': {
            fontSize: '1rem',
            fontWeight: 600,
            textTransform: 'none',
            py: 2
          }
        }}
      >
        <Tab 
          icon={<School />} 
          iconPosition="start" 
          label="Program Success" 
          value="success" 
        />
        <Tab 
          icon={<TrendingUp />} 
          iconPosition="start" 
          label="Persistence" 
          value="persistence" 
        />
        <Tab 
          icon={<Assessment />} 
          iconPosition="start" 
          label="GPA Prediction" 
          value="gpa" 
        />
      </Tabs>
    </Box>
  );
};

export default PredictionTabs;