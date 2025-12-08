import React from 'react';
import { Card, CardContent, Typography, Box, Divider, Chip } from '@mui/material';
import { CheckCircle, Warning, TrendingUp } from '@mui/icons-material';

const ResultCard = ({ title, data, type = 'success' }) => {
  const getColor = () => {
    if (type === 'gpa') return '#667eea';
    return data.prediction?.includes('Complete') || 
           data.prediction?.includes('Will Persist') || 
           data.risk_level === 'Low' 
      ? '#4caf50' 
      : '#ff9800';
  };

  const getIcon = () => {
    if (type === 'gpa') return <TrendingUp sx={{ fontSize: 40 }} />;
    return data.prediction?.includes('Complete') || 
           data.prediction?.includes('Will Persist') 
      ? <CheckCircle sx={{ fontSize: 40 }} /> 
      : <Warning sx={{ fontSize: 40 }} />;
  };

  return (
    <Card 
      sx={{ 
        mt: 3,
        borderLeft: `5px solid ${getColor()}`,
        boxShadow: 3,
        animation: 'slideIn 0.5s ease'
      }}
    >
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
          <Box sx={{ color: getColor() }}>
            {getIcon()}
          </Box>
          <Typography variant="h5" fontWeight="bold">
            {title}
          </Typography>
        </Box>

        <Divider sx={{ mb: 2 }} />

        {/* Success/Persistence Results */}
        {(type === 'success' || type === 'persistence') && (
          <>
            <ResultRow 
              label="Outcome" 
              value={data.prediction}
              highlight
            />
            <ResultRow 
              label="Probability" 
              value={`${(data.probability * 100).toFixed(1)}%`}
            />
            {data.confidence && (
              <ResultRow 
                label="Confidence" 
                value={`${(data.confidence * 100).toFixed(1)}%`}
              />
            )}
            {data.risk_level && (
              <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="body1" fontWeight="600">
                  Risk Level:
                </Typography>
                <Chip 
                  label={data.risk_level}
                  color={
                    data.risk_level === 'Low' ? 'success' : 
                    data.risk_level === 'Medium' ? 'warning' : 'error'
                  }
                  sx={{ fontWeight: 'bold' }}
                />
              </Box>
            )}
          </>
        )}

        {/* GPA Results */}
        {type === 'gpa' && (
          <>
            <Box sx={{ textAlign: 'center', my: 3 }}>
              <Typography variant="h2" fontWeight="bold" color="primary">
                {data.predicted_gpa}
              </Typography>
              <Typography variant="h6" color="text.secondary">
                Predicted GPA
              </Typography>
            </Box>
            <Divider sx={{ my: 2 }} />
            <ResultRow 
              label="Letter Grade" 
              value={data.grade_letter}
            />
            <ResultRow 
              label="Expected Change" 
              value={`${data.improvement >= 0 ? '+' : ''}${data.improvement}`}
              valueColor={data.improvement >= 0 ? 'success.main' : 'error.main'}
            />
          </>
        )}
      </CardContent>
    </Card>
  );
};

const ResultRow = ({ label, value, highlight = false, valueColor = 'text.primary' }) => (
  <Box 
    sx={{ 
      display: 'flex', 
      justifyContent: 'space-between', 
      alignItems: 'center',
      py: 1.5,
      borderBottom: '1px solid',
      borderColor: 'divider'
    }}
  >
    <Typography variant="body1" fontWeight="600" color="text.secondary">
      {label}:
    </Typography>
    <Typography 
      variant="body1" 
      fontWeight={highlight ? 'bold' : 'normal'}
      sx={{ 
        fontSize: highlight ? '1.2rem' : '1rem',
        color: valueColor
      }}
    >
      {value}
    </Typography>
  </Box>
);

export default ResultCard;