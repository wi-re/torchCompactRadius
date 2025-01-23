import os
for subdir in os.listdir('wheels'):
    if not os.path.isdir(f'./wheels/{subdir}'):
        continue
    print(subdir)

    wheels = []
    for wheel in os.listdir(f'./wheels/{subdir}'):
        file = f'wheel/{subdir}/{wheel}'
        if file.endswith('.whl'):
            wheels.append(wheel)
    # print(wheels)
    with open(f'./wheels/{subdir}/index.html', 'w') as f:
        f.write('<!DOCTYPE html>\n')
        f.write('<html>\n<body>\n')
        for wheel in wheels:
            f.write(f'<a href="./{wheel}">{wheel}</a><br>\n')
        f.write('</body>\n</html>')

    # break
