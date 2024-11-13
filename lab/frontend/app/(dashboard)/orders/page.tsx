import * as React from 'react';
import Typography from '@mui/material/Typography';
import { Suspense } from 'react';

function OrdersContent() {
  return (
    <Typography>
      Welcome to the Toolpad orders!
    </Typography>
  );
}

export default function OrdersPage() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <OrdersContent />
    </Suspense>
  );
}