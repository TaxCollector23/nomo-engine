import RunDashboard from "@/components/RunDashboard";

export default function RunPage({ params }: { params: { runId: string } }) {
  return <RunDashboard runId={params.runId} />;
}
