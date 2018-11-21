function [q] = myeul2quat(r, p, y)
	cy = cos(y * 0.5);
	sy = sin(y * 0.5);
	cr = cos(r * 0.5);
	sr = sin(r * 0.5);
	cp = cos(p * 0.5);
	sp = sin(p * 0.5);

    q = zeros(4,1);
	q(1) = cy * cr * cp + sy * sr * sp;
	q(2) = cy * sr * cp - sy * cr * sp;
	q(3) = cy * cr * sp + sy * sr * cp;
	q(4) = sy * cr * cp - cy * sr * sp;
end