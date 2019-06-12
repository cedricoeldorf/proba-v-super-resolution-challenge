import sys
import math
from itertools import islice

radius = 1

def get_line_number(phrase, file_name):
    with open(file_name) as f:
        for i, line in enumerate(f, 1):
            if phrase in line:
                return i
        return False

def read_weights(file_name, ln, size=1):
    content = []
    with open(file_name) as f:
        for line in islice(f, ln, ln + size):
            if line.find('[') != -1:
                line = line[line.index('[') + 1:]
            if line.find(']') != -1:
                line = line[:line.rindex(']')]
            content.append(line)

    return [x.strip() for x in content]

def format_weights(weights, n, length=4):
    return ",".join(['{:.16f}'.format(float(i)) for i in weights.strip(",").split(",")[n:n+length]])

def base_header(file):
    file.write('//!HOOK LUMA\n')
    file.write('//!WHEN OUTPUT.w LUMA.w / {0}.400 > OUTPUT.h LUMA.h / {0}.400 > *\n'.format(scale - 1))

def header1(file, n, d):
    base_header(file)
    file.write('//!DESC feature map {}\n'.format((n//4)%(d//4) + 1))
    file.write('//!BIND LUMA\n')
    file.write('//!SAVE FEATURE{}\n'.format((n//4)%(d//4) + 1))
    file.write('//!COMPONENTS 4\n')

def header2(file, d, n, s):
    base_header(file)
    file.write('//!DESC shrinking {}\n'.format((n//4)%(s//4) + 1))
    for i in range(d//4):
        file.write('//!BIND {}{}\n'.format("FEATURE", i + 1))
    file.write('//!SAVE SHRINKED{}\n'.format((n//4)%(s//4) + 1))
    file.write('//!COMPONENTS 4\n')

def header3(file, r, mi, m, n, s, inp):
    base_header(file)
    file.write('//!DESC mapping {}_{}\n'.format(mi + 1, (n//4)%(s//4) + 1))
    for i in range(s//4):
        file.write('//!BIND {}{}\n'.format(inp, i+1 + (0 if (r * m + mi) % 2 == 0 else 20)))
    file.write('//!SAVE MODEL{}\n'.format((n//4)%(s//4) + 1 + (20 if (r * m + mi) % 2 == 0 else 0)))
    file.write('//!COMPONENTS 4\n')

def header3_1(file, r, mi, m, n, s, inp):
    base_header(file)
    file.write('//!DESC sub-band residuals {}\n'.format((n//4)%(s//4) + 1))
    for i in range(s//4):
        file.write('//!BIND MODEL{}\n'.format(i + 1 + (20 if (r * m + mi) % 2 == 0 else 0)))
    file.write('//!BIND {}{}\n'.format(inp, (n//4)%(s//4) + 1))
    file.write('//!SAVE RES{}\n'.format((n//4)%(s//4) + 1))
    file.write('//!COMPONENTS 4\n')

def header4(file, s, m, r, n, d):
    base_header(file)
    file.write('//!DESC expanding {}\n'.format((n//4)%(d//4) + 1))
    for i in range(s//4):
        file.write('//!BIND RES{}\n'.format(i + 1))
    file.write('//!SAVE EXPANDED{}\n'.format((n//4)%(d//4) + 1))
    file.write('//!COMPONENTS 4\n')

def header5(file, n, d, inp):
    base_header(file)
    file.write('//!DESC sub-pixel convolution {}\n'.format((n//comps) + 1))
    for i in range(d//4):
        file.write('//!BIND {}{}\n'.format(inp, i + 1))
    file.write('//!SAVE SUBCONV{}\n'.format((n//comps) + 1))
    file.write('//!COMPONENTS {}\n'.format(comps))

def header6(file):
    base_header(file)
    file.write('//!WIDTH LUMA.w {} *\n'.format(scale))
    file.write('//!HEIGHT LUMA.h {} *\n'.format(scale))
    file.write('//!DESC aggregation\n')
    for i in range(scale**2//comps):
        file.write('//!BIND SUBCONV{}\n'.format(i + 1))

def main():
  if len(sys.argv) == 2:
    fname=sys.argv[1]
    d, s, m, r = [int(i) for i in fname[7:fname.index('.')].split("_")]
    if s == 0:
        s = d
        shrinking = False
    else:
        shrinking = True
    global scale, comps
    deconv_biases = read_weights(fname, get_line_number("deconv_b", fname))
    scale = int(math.sqrt(len(deconv_biases[0].split(","))))
    dst = fname.replace("_", "-").replace("weights", "FSRCNNX_x{}_".format(scale)).replace("txt", "glsl")
    with open(dst, 'w') as file:

        # Feature layer
        feature_radius = 2
        ln = get_line_number("w1", fname)
        weights = read_weights(fname, ln, (feature_radius*2+1)**2)
        ln = get_line_number("b1", fname)
        biases = read_weights(fname, ln)
        for n in range(0, d, 4):
            header1(file, n, d)
            file.write('vec4 hook()\n')
            file.write('{\n')
            file.write('vec4 res = vec4({});\n'.format(format_weights(biases[0], n)))
            p = 0
            for l in range(0, len(weights)):
                y, x = p%(feature_radius*2+1)-feature_radius, p//(feature_radius*2+1)-feature_radius
                p += 1
                file.write('res += vec4({}) * float(LUMA_texOff(vec2({},{})));\n'.format(format_weights(weights[l], n), x, y))
            if shrinking:
                ln = get_line_number("alpha1", fname)
                alphas = read_weights(fname, ln)
                file.write('res = max(res, vec4(0.0)) + vec4({}) * min(res, vec4(0.0));\n'.format(format_weights(alphas[0], n)))
            file.write('return res;\n')
            file.write('}\n\n')

        if shrinking:
            # Shrinking layer
            ln = get_line_number("w2", fname)
            weights = read_weights(fname, ln, d)
            ln = get_line_number("b2", fname)
            biases = read_weights(fname, ln)
            for n in range(0, s, 4):
                header2(file, d, n, s)
                file.write('vec4 hook()\n')
                file.write('{\n')
                file.write('vec4 res = vec4({});\n'.format(format_weights(biases[0], n)))
                for l in range(0, d, 4):
                    file.write('res += mat4({},{},{},{}) * FEATURE{}_texOff(vec2(0.0));\n'.format(format_weights(weights[l], n), format_weights(weights[l+1], n), format_weights(weights[l+2], n), format_weights(weights[l+3], n), l//4+1))
                file.write('return res;\n')
                file.write('}\n\n')

        # Mapping layers
        inp = "SHRINKED" if shrinking else "FEATURE"
        for ri in range(r):
            for mi in range(m):
                tex_name = inp if ri == 0 and mi == 0 else "RES" if ri > 0 and mi == 0 else "MODEL"
                ln = get_line_number("w{}".format(mi + 3), fname)
                weights = read_weights(fname, ln, s*9)
                ln = get_line_number("b{}".format(mi + 3), fname)
                biases = read_weights(fname, ln)
                for n in range(0, s, 4):
                    header3(file, ri, mi, m, n, s, tex_name)
                    file.write('vec4 hook()\n')
                    file.write('{\n')
                    file.write('vec4 res = vec4({});\n'.format(format_weights(biases[0], n)))
                    p = 0
                    for l in range(0, len(weights), 4):
                        if l % s == 0:
                            y, x = p%3-1, p//3-1
                            p += 1
                        idx = (l//4)%(s//4)
                        file.write('res += mat4({},{},{},{}) * {}{}_texOff(vec2({},{}));\n'.format(
                                    format_weights(weights[l], n), format_weights(weights[l+1], n),
                                    format_weights(weights[l+2], n), format_weights(weights[l+3], n),
                                    tex_name, idx + 1 + (20 if (ri * m + mi) % 2 == 1 else 0), x, y))
                    ln = get_line_number("alpha{}".format(m + 3 if mi == m - 1 else mi + 4), fname)
                    alphas = read_weights(fname, ln)
                    file.write('res = max(res, vec4(0.0)) + vec4({}) * min(res, vec4(0.0));\n'.format(format_weights(alphas[0], n)))
                    file.write('return res;\n')
                    file.write('}\n\n')

                if mi == m - 1:
                    ln = get_line_number("w{}".format(m + 3), fname)
                    weights = read_weights(fname, ln, s*(mi+2))
                    ln = get_line_number("b{}".format(m + 3), fname)
                    biases = read_weights(fname, ln)
                    for n in range(0, s, 4):
                        header3_1(file, ri, mi, m, n, s, inp)
                        file.write('vec4 hook()\n')
                        file.write('{\n')
                        file.write('vec4 res = vec4({});\n'.format(format_weights(biases[0], n)))
                        for l in range(0, s, 4):
                            file.write('res += mat4({},{},{},{}) * MODEL{}_texOff(0);\n'.format(
                                       format_weights(weights[l], n), format_weights(weights[l+1], n),
                                       format_weights(weights[l+2], n), format_weights(weights[l+3], n),
                                       l//4 + 1 + (20 if (ri * m + mi) % 2 == 0 else 0)))
                        file.write('res += {}{}_texOff(0);\n'.format(inp, (n//4)%(s//4) + 1))
                        if ri == r - 1:
                            ln = get_line_number("alpha2", fname)
                            alphas = read_weights(fname, ln)
                            file.write('res = max(res, vec4(0.0)) + vec4({}) * min(res, vec4(0.0));\n'.format(format_weights(alphas[0], n)))
                        file.write('return res;\n')
                        file.write('}\n\n')

        if shrinking:
            # Expanding layer
            ln = get_line_number("w{}".format(m + 4), fname)
            weights = read_weights(fname, ln, d)
            ln = get_line_number("b{}".format(m + 4), fname)
            biases = read_weights(fname, ln)
            ln = get_line_number("alpha{}".format(m + 4), fname)
            alphas = read_weights(fname, ln)
            for n in range(0, d, 4):
                header4(file, s, m, r, n, d)
                file.write('vec4 hook()\n')
                file.write('{\n')
                file.write('vec4 res = vec4({});\n'.format(format_weights(biases[0], n)))
                for l in range(0, s, 4):
                    file.write('res += mat4({},{},{},{}) * RES{}_texOff(vec2(0.0));\n'.format(format_weights(weights[l], n), format_weights(weights[l+1], n), format_weights(weights[l+2], n), format_weights(weights[l+3], n),
                    l//4 + 1))
                file.write('res = max(res, vec4(0.0)) + vec4({}) * min(res, vec4(0.0));\n'.format(format_weights(alphas[0], n)))
                file.write('return res;\n')
                file.write('}\n\n')

        # Sub-pixel convolution
        ln = get_line_number("deconv_w", fname)
        weights = read_weights(fname, ln, d*(radius*2+1)**2)
        ln = get_line_number("deconv_b", fname)
        biases = read_weights(fname, ln)
        inp = "EXPANDED" if shrinking else "RES"
        comps = 3 if scale == 3 else 4
        for n in range(0, scale**2, comps):
            header5(file, n, d, inp)
            file.write('vec4 hook()\n')
            file.write('{\n')
            file.write('vec{0} res = vec{0}({1});\n'.format(comps, format_weights(biases[0], n, length=comps)))
            p = 0
            for l in range(0, len(weights), 4):
                if l % d == 0:
                    y, x = p%(radius*2+1)-radius, p//(radius*2+1)-radius
                    p += 1
                idx = (l//4)%(d//4)
                file.write('res += mat4x{}({},{},{},{}) * {}{}_texOff(vec2({},{}));\n'.format(
                           comps, format_weights(weights[l], n, length=comps), format_weights(weights[l+1], n, length=comps),
                           format_weights(weights[l+2], n, length=comps), format_weights(weights[l+3], n, length=comps),
                           inp, idx + 1, x, y))
            if comps == 4:
                file.write('return res;\n')
            else:
                file.write('return vec4(res, 0);\n')
            file.write('}\n\n')

        # Aggregation
        header6(file)
        file.write('vec4 hook()\n')
        file.write('{\n')
        file.write('vec2 fcoord = fract(SUBCONV1_pos * SUBCONV1_size);\n')
        file.write('vec2 base = SUBCONV1_pos + (vec2(0.5) - fcoord) * SUBCONV1_pt;\n')
        file.write('ivec2 index = ivec2(fcoord * vec2({}));\n'.format(scale))
        if scale > 2:
            file.write('mat{0} res = mat{0}(SUBCONV1_tex(base).{1}'.format(scale, "rgba"[:comps]))
            for i in range(scale-1):
                file.write(',SUBCONV{}_tex(base).{}'.format(i + 2, "rgba"[:comps]))
            file.write(');\n')
            file.write('return vec4(res[index.x][index.y], 0, 0, 1);\n')
        else:
            file.write('vec4 res = SUBCONV1_tex(base);\n')
            file.write('return vec4(res[index.x * {} + index.y], 0, 0, 1);\n'.format(scale))
        file.write('}\n')

  else:
    print("Missing argument: You must specify a file name")
    return

if __name__ == '__main__':
  main()
