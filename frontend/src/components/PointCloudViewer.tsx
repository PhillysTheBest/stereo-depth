import { OrbitControls } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { useEffect, useMemo } from "react";
import { BufferAttribute, BufferGeometry } from "three";

import { bounds, type PointCloud } from "../three/pointCloud";

interface Props {
  cloud: PointCloud;
  /** Point size in scene units. Defaults to a fraction of the cloud's extent. */
  pointSize?: number;
  height?: number;
}

/** Orbitable point cloud. Starts from the left camera's viewpoint, looking at the cloud's centre. */
export function PointCloudViewer({ cloud, pointSize, height = 480 }: Props) {
  const geometry = useMemo(() => {
    const g = new BufferGeometry();
    g.setAttribute("position", new BufferAttribute(cloud.positions, 3));
    g.setAttribute("color", new BufferAttribute(cloud.colors, 3));
    return g;
  }, [cloud]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  const { centre, size } = useMemo(() => bounds(cloud), [cloud]);

  return (
    <div role="img" aria-label={`3D point cloud, ${cloud.count} points`} style={{ height }}>
      <Canvas camera={{ position: [0, 0, 0], fov: 50, near: size / 1000, far: size * 100 }}>
        <color attach="background" args={["#111"]} />
        <points geometry={geometry}>
          <pointsMaterial size={pointSize ?? size / 400} vertexColors sizeAttenuation />
        </points>
        <OrbitControls target={centre} makeDefault />
      </Canvas>
    </div>
  );
}
