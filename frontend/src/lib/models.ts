/** Friendly names for the built-in example models. */
export const MODEL_BLURB: Record<string, { title: string; body: string }> = {
  perception_cnn: { title: "Event-camera vision", body: "A 7-layer image network that recognises 11 gestures from an event camera." },
  attitude_policy: { title: "Drone attitude controller", body: "A 7-layer controller that turns sensor readings into motor torques, with a physics model and safety limits." },
};

export function modelTitle(id: string, uploadedName?: string): string {
  return MODEL_BLURB[id]?.title ?? uploadedName ?? id;
}
