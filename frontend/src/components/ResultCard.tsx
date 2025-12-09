import React from 'react';
import { Card, CardContent, Typography, Box, Divider, Chip } from '@mui/material';
import { CheckCircle, Warning as WarningIcon, TrendingUp } from '@mui/icons-material';

type TabType = 'persistence' | 'dropout' | 'gpa' | 'success';

interface ResultCardProps {
  title: string;
  data: any;
  type: TabType;
}

const ResultCard: React.FC<ResultCardProps> = ({ title, data, type }) => {
  const getColor = () => {
    if (type === 'gpa') return 'primary';
    if (type === 'dropout') {
      return data.prediction?.includes('Will Continue') ? 'success' : 'warning';
    }
    return data.prediction?.includes('Complete') || 
           data.prediction?.includes('Will Persist') || 
           data.risk_level === 'Low' 
      ? 'success' 
      : 'warning';
  };

  const getIcon = () => {
    if (type === 'gpa') return <TrendingUp sx={{ fontSize: 40 }} />;
    if (type === 'dropout') {
      return data.prediction?.includes('Will Continue') 
        ? <CheckCircle sx={{ fontSize: 40 }} /> 
        : <WarningIcon sx={{ fontSize: 40 }} />;
    }
    return data.prediction?.includes('Complete') || 
           data.prediction?.includes('Will Persist') 
      ? <CheckCircle sx={{ fontSize: 40 }} /> 
      : <WarningIcon sx={{ fontSize: 40 }} />;
  };

  const ResultRow = ({ label, value, highlight = false }: any) => (
    <Box sx={{ display: 'flex', justifyContent: 'space-between', py: 1.5, borderBottom: '1px solid', borderColor: 'divider' }}>
      <Typography variant="body1" fontWeight="600" color="text.secondary">{label}:</Typography>
      <Typography variant="body1" fontWeight={highlight ? 'bold' : 'normal'} sx={{ fontSize: highlight ? '1.2rem' : '1rem' }}>
        {value}
      </Typography>
    </Box>
  );

  return (
    <Card sx={{ mt: 3, borderLeft: 5, borderColor: `${getColor()}.main`, boxShadow: 3 }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
          <Box sx={{ color: `${getColor()}.main` }}>{getIcon()}</Box>
          <Typography variant="h5" fontWeight="bold">{title}</Typography>
        </Box>
        <Divider sx={{ mb: 2 }} />

        {type === 'gpa' ? (
          <>
            <Box sx={{ textAlign: 'center', my: 3 }}>
              <Typography variant="h2" fontWeight="bold" color="primary">{data.predicted_gpa}</Typography>
              <Typography variant="h6" color="text.secondary">Predicted GPA</Typography>
            </Box>
            <Divider sx={{ my: 2 }} />
            <ResultRow label="Letter Grade" value={data.grade_letter} />
            <ResultRow label="Expected Change" value={`${data.improvement >= 0 ? '+' : ''}${data.improvement}`} />
            <ResultRow label="Trend" value={data.trend} />
            {data.recommendation && (
              <Box sx={{ mt: 2, p: 2, bgcolor: 'info.light', borderRadius: 1 }}>
                <Typography variant="body2">{data.recommendation}</Typography>
              </Box>
            )}
          </>
        ) : (
          <>
            <ResultRow label="Outcome" value={data.prediction} highlight />
            <ResultRow label="Probability" value={`${((data.probability || data.dropout_probability || 0) * 100).toFixed(1)}%`} />
            {data.confidence && <ResultRow label="Confidence" value={`${(data.confidence * 100).toFixed(1)}%`} />}
            {data.confidence_level && <ResultRow label="Confidence Level" value={data.confidence_level} />}
            {data.risk_level && (
              <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="body1" fontWeight="600">Risk Level:</Typography>
                <Chip 
                  label={data.risk_level} 
                  color={data.risk_level === 'Low' ? 'success' : data.risk_level === 'Medium' ? 'warning' : 'error'} 
                  sx={{ fontWeight: 'bold' }} 
                />
              </Box>
            )}
            {data.recommendation && (
              <Box sx={{ mt: 2, p: 2, bgcolor: 'info.light', borderRadius: 1 }}>
                <Typography variant="body2" fontWeight="500">Recommendation:</Typography>
                <Typography variant="body2">{data.recommendation}</Typography>
              </Box>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
};



export default ResultCard;