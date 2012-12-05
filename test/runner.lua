#!/usr/bin/env torch

package.path = './?/init.lua;' .. package.path

require 'paths'

local function findTestFiles()
    local testFiles = {}
    for file in paths.files("test/") do
        if string.match(file, "^test_.*%.lua$") then
            table.insert(testFiles, "test/" .. file)
        end
    end
    return testFiles
end

local function main()
    local testFiles = findTestFiles()
    for i, file in ipairs(testFiles) do
        print("Doing:", file)
        _G.tests = {}
        _G.tester = torch.Tester()
        dofile(file)

        tester:add(tests)
        tester:run()
        if #tester.errors > 0 then
            os.exit(1)
        end
    end
end

main()
