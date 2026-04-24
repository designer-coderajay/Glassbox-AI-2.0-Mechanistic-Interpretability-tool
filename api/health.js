export const config = {
  runtime: 'edge',
};

export default async function handler(request) {
  if (request.method !== 'GET' && request.method !== 'HEAD') {
    return new Response(
      JSON.stringify({ success: false, error: 'Method not allowed.' }),
      {
        status: 405,
        headers: { 'Content-Type': 'application/json' },
      },
    );
  }

  return new Response(
    JSON.stringify({
      status: 'ok',
      service: 'glassbox-api',
      timestamp: new Date().toISOString(),
    }),
    {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-store',
      },
    },
  );
}
